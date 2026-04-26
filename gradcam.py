import numpy as np
import cv2
import os
import tensorflow as tf
from typing import Optional


# ================================================================
#  CONFIGURATION — change values here without touching other code
# ================================================================
GRADCAM_CONFIG = {
    "percentile_low":   2,           # clip bottom 2%
    "percentile_high":  98,          # clip top   98%
    "gaussian_blur":    9,           # smooth kernel (odd number only)
    "default_alpha":    0.55,        # heatmap blend strength (0=invisible, 1=opaque)
    "colormap":         cv2.COLORMAP_JET,
    "colorbar_width":   35,
    "colorbar_pad":     8,
    "gamma":            0.5,         # lowered from 0.6 → more aggressive brightening
    "contour_thresh":   0.60,        # slightly lowered to show more contour regions
}


# ================================================================
#  AUTOMATIC CONV LAYER DISCOVERY
# ================================================================
def find_last_conv_layer(model) -> str:
    """
    Walk model layers in reverse and return the name of the last Conv2D.
    Handles both flat and nested (functional) model structures.
    """
    # First try top-level layers
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            print(f"[GradCAM] Auto-selected conv layer (top-level): '{layer.name}'")
            return layer.name

    # If not found, search inside nested sub-models
    for layer in reversed(model.layers):
        if hasattr(layer, 'layers'):
            for sublayer in reversed(layer.layers):
                if isinstance(sublayer, tf.keras.layers.Conv2D):
                    print(f"[GradCAM] Auto-selected conv layer (nested): '{sublayer.name}'")
                    return sublayer.name

    raise ValueError(
        "[GradCAM] No Conv2D layer found in model.\n"
        "Run model.summary() to check your architecture."
    )


# ================================================================
#  STEP 1 — Normalize raw heatmap to clean 0-1 range
# ================================================================
def _normalize_heatmap(raw: np.ndarray) -> np.ndarray:
    """
    Normalize raw Grad-CAM activations to [0, 1] float32.

    Pipeline
    --------
    ReLU → Percentile Clip → Min-Max Scale → Full Range Stretch
    → Gaussian Blur → Gamma Boost

    Key fixes vs previous version
    ------------------------------
    1. Added full range stretch: heatmap / heatmap.max()
       Forces the brightest pixel to exactly 1.0 (pure red in JET).
       Without this, max was stuck at ~0.66, meaning no red ever appeared.

    2. Gamma lowered to 0.5 for more aggressive mid-tone brightening.
    """

    # --- ReLU: keep only positive contributions ---
    heatmap = np.maximum(raw, 0).astype(np.float32)

    # --- Guard: all-zero heatmap (dead gradients) ---
    if heatmap.max() < 1e-8:
        print(
            "[GradCAM] WARNING: All-zero heatmap received by _normalize_heatmap.\n"
            "          Likely cause: GradientTape did not capture gradients.\n"
            "          Check that tape.watch() wraps the conv_outputs tensor\n"
            "          BEFORE the class_channel computation."
        )
        return np.zeros_like(heatmap)

    # --- Percentile clip: stretch contrast ---
    p_low  = np.percentile(heatmap, GRADCAM_CONFIG["percentile_low"])
    p_high = np.percentile(heatmap, GRADCAM_CONFIG["percentile_high"])

    print(f"[GradCAM] Percentile clip — p{GRADCAM_CONFIG['percentile_low']}: {p_low:.6f}, "
          f"p{GRADCAM_CONFIG['percentile_high']}: {p_high:.6f}")

    if p_high - p_low < 1e-6:
        print(
            "[GradCAM] WARNING: Flat activation map detected.\n"
            "          Model gradients are near zero.\n"
            "          Verify last_conv_layer_name with model.summary()."
        )
        return np.zeros_like(heatmap)

    heatmap = np.clip(heatmap, p_low, p_high)

    # --- Min-Max normalise to [0, 1] ---
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # --- ✅ FIX: Force full dynamic range ---
    # Without this, max was stuck at ~0.66 → no red in JET colormap
    # This ensures the brightest pixel always maps to pure red (255 in JET)
    if heatmap.max() > 1e-8:
        heatmap = heatmap / heatmap.max()

    print(f"[GradCAM] Normalised heatmap — min: {heatmap.min():.4f}, max: {heatmap.max():.4f}")

    # --- Gaussian smoothing ---
    k      = GRADCAM_CONFIG["gaussian_blur"]
    h_size = min(heatmap.shape[0], heatmap.shape[1])
    k      = min(k, h_size)
    if k % 2 == 0:
        k -= 1
    k = max(k, 1)
    heatmap = cv2.GaussianBlur(heatmap, (k, k), sigmaX=0)

    # --- Gamma correction (γ < 1 → brightens mid-tones → more red/yellow) ---
    heatmap = np.power(heatmap, GRADCAM_CONFIG["gamma"]).astype(np.float32)

    return heatmap


# ================================================================
#  STEP 2 — Compute Grad-CAM heatmap from model
# ================================================================
def get_gradcam_heatmap(
    model,
    img_array:            np.ndarray,
    clinical_data:        np.ndarray,
    last_conv_layer_name: str = "auto",
) -> np.ndarray:
    """
    Generate Grad-CAM heatmap.

    Key fixes in this version
    -------------------------
    1. training=True in grad_model() call
       BatchNormalization with training=False freezes statistics and causes
       gradients to vanish for many inputs (the all-zero heatmap warning).
       Using training=True keeps BatchNorm active so gradients flow correctly.

    2. tape.watch(img_tensor) placed BEFORE the forward pass.
       GradientTape records operations only for tensors it watches at the
       time those operations execute. Watching after the forward pass means
       the tape has no record of how conv_outputs was computed.

    3. Auto layer detection handles both flat and nested model structures.
    """
    try:
        # ── Auto-detect last conv layer ───────────────────────────────────
        if last_conv_layer_name in ("auto", "conv2d"):
            last_conv_layer_name = find_last_conv_layer(model)

        print(f"[GradCAM] Using conv layer: '{last_conv_layer_name}'")

        # ── Build sub-model: conv features + predictions ──────────────────
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[
                model.get_layer(last_conv_layer_name).output,
                model.output,
            ]
        )

        # Cast inputs
        img_tensor      = tf.cast(img_array,     tf.float32)
        clinical_tensor = tf.cast(clinical_data, tf.float32)

        # ── GradientTape ──────────────────────────────────────────────────
        with tf.GradientTape() as tape:

            # Watch BEFORE forward pass
            tape.watch(img_tensor)

            # ✅ FIX: training=True — keeps BatchNorm active
            # training=False was causing gradients to vanish on many images
            conv_outputs, predictions = grad_model(
                [img_tensor, clinical_tensor], training=True
            )

            pred_index    = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

            print(f"[GradCAM] Predicted class index: {pred_index.numpy()}")

        # ── Compute gradients ─────────────────────────────────────────────
        grads = tape.gradient(class_channel, conv_outputs)

        if grads is None:
            raise ValueError(
                f"[GradCAM] Gradients are None for layer '{last_conv_layer_name}'.\n"
                "Reasons: wrong layer name, disconnected layer, or non-differentiable ops."
            )

        print(f"[GradCAM] Gradient shape: {grads.shape}, "
              f"min: {tf.reduce_min(grads):.6f}, max: {tf.reduce_max(grads):.6f}")

        # ── Global Average Pool → importance weight per filter ────────────
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        non_zero = tf.math.count_nonzero(pooled_grads).numpy()
        print(f"[GradCAM] Pooled grad shape: {pooled_grads.shape}, "
              f"non-zero: {non_zero}/{pooled_grads.shape[0]}")

        # ── Weighted sum of feature maps ──────────────────────────────────
        conv_map = conv_outputs[0]                             # (h, w, filters)
        heatmap  = conv_map @ pooled_grads[..., tf.newaxis]   # (h, w, 1)
        heatmap  = tf.squeeze(heatmap).numpy()                 # (h, w)

        print(f"[GradCAM] Raw heatmap shape: {heatmap.shape}, "
              f"min: {heatmap.min():.6f}, max: {heatmap.max():.6f}")

        # Post-ReLU stats
        relu_map = np.maximum(heatmap, 0)
        print(f"[GradCAM] Post-ReLU heatmap — "
              f"min: {relu_map.min():.6f}, max: {relu_map.max():.6f}, "
              f"mean: {relu_map.mean():.6f}")

        # ── Normalize ─────────────────────────────────────────────────────
        heatmap = _normalize_heatmap(heatmap)

        return heatmap

    except Exception as exc:
        print(f"[GradCAM] Heatmap generation failed: {exc}")
        return np.zeros((8, 8), dtype=np.float32)


# ================================================================
#  STEP 3 — Superimpose heatmap on original MRI image
# ================================================================
def superimpose_heatmap(
    image_path: str,
    heatmap:    np.ndarray,
    intensity:  float = GRADCAM_CONFIG["default_alpha"],
    save_dir:   str   = "uploads",
) -> Optional[str]:
    """
    Overlay Grad-CAM heatmap on original MRI with CLAHE contrast
    enhancement, attention contours, and colorbar legend.
    """
    try:
        orig = cv2.imread(image_path)
        if orig is None:
            raise FileNotFoundError(f"[GradCAM] Cannot read image at: {image_path}")

        h, w  = orig.shape[:2]
        alpha = float(np.clip(intensity, 0.0, 1.0))

        # ── Enhance original MRI contrast ─────────────────────────────────
        orig_enhanced = _apply_clahe(orig)

        # ── Resize heatmap to MRI resolution ─────────────────────────────
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)

        # ── Boost heatmap contrast ────────────────────────────────────────
        heatmap_boosted = _enhance_heatmap_contrast(heatmap_resized)

        # ── Convert to uint8 and apply JET colormap ───────────────────────
        heatmap_uint8 = np.uint8(255 * heatmap_boosted)
        colormap_img  = cv2.applyColorMap(heatmap_uint8, GRADCAM_CONFIG["colormap"])

        # ── Alpha blend ───────────────────────────────────────────────────
        superimposed = cv2.addWeighted(
            colormap_img,   alpha,
            orig_enhanced,  1.0 - alpha,
            0
        )

        # ── Attention contours ────────────────────────────────────────────
        superimposed = _add_attention_contour(superimposed, heatmap_resized)

        # ── Colorbar legend ───────────────────────────────────────────────
        superimposed = _add_colorbar(superimposed)

        # ── Save ──────────────────────────────────────────────────────────
        os.makedirs(save_dir, exist_ok=True)
        base     = os.path.splitext(os.path.basename(image_path))[0]
        out_name = f"gradcam_{base}_a{int(alpha * 100):03d}.jpg"
        out_path = os.path.join(save_dir, out_name)
        cv2.imwrite(out_path, superimposed, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"[GradCAM] Heatmap saved → {out_path}")

        return out_path

    except Exception as exc:
        print(f"[GradCAM] Superimpose failed: {exc}")
        return None


# ================================================================
#  INTERNAL — CLAHE contrast on original MRI
# ================================================================
def _apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE to each BGR channel independently.
    Keeps brain anatomy clearly visible under the heatmap overlay.
    """
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    channels = cv2.split(image)
    enhanced = cv2.merge([clahe.apply(c) for c in channels])
    return enhanced


# ================================================================
#  INTERNAL — Boost heatmap contrast to push more red/yellow
# ================================================================
def _enhance_heatmap_contrast(heatmap: np.ndarray) -> np.ndarray:
    """
    Second-pass contrast boost using CLAHE on the heatmap.
    Pushes activations into the red/yellow zone of JET colormap.
    """
    h8    = np.uint8(255 * heatmap)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    h8    = clahe.apply(h8)
    return h8.astype(np.float32) / 255.0


# ================================================================
#  INTERNAL — Draw attention contours
# ================================================================
def _add_attention_contour(
    image:     np.ndarray,
    heatmap:   np.ndarray,
    threshold: float = GRADCAM_CONFIG["contour_thresh"],
) -> np.ndarray:
    """
    Draw thin white contours around high-attention brain regions.
    Mimics contour overlays used in IEEE/medical paper Grad-CAM figures.
    """
    try:
        binary      = np.uint8(heatmap >= threshold) * 255
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        overlay = image.copy()
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)
        return cv2.addWeighted(overlay, 0.4, image, 0.6, 0)
    except Exception:
        return image


# ================================================================
#  INTERNAL — Vertical JET colorbar with labels
# ================================================================
def _add_colorbar(image: np.ndarray) -> np.ndarray:
    """
    Append a vertical JET colorbar legend on the right side of the image.
    HIGH (red, top) → MED (yellow-green) → LOW (blue, bottom)
    """
    h   = image.shape[0]
    bw  = GRADCAM_CONFIG["colorbar_width"]
    pad = GRADCAM_CONFIG["colorbar_pad"]

    bar = np.zeros((h, bw, 3), dtype=np.uint8)
    for row in range(h):
        val   = int(255 * (1.0 - row / h))
        color = cv2.applyColorMap(
            np.array([[val]], dtype=np.uint8),
            GRADCAM_CONFIG["colormap"]
        )[0][0]
        bar[row, :] = color

    font = cv2.FONT_HERSHEY_SIMPLEX
    clr  = (255, 255, 255)
    cv2.putText(bar, "HI",  (2, 12),        font, 0.33, clr, 1, cv2.LINE_AA)
    cv2.putText(bar, "MED", (2, h // 2),    font, 0.33, clr, 1, cv2.LINE_AA)
    cv2.putText(bar, "LO",  (2, h - 6),     font, 0.33, clr, 1, cv2.LINE_AA)
    cv2.rectangle(bar, (0, 0), (bw - 1, h - 1), (180, 180, 180), 1)

    sep = np.full((h, pad, 3), 15, dtype=np.uint8)
    return np.concatenate([image, sep, bar], axis=1)


# ================================================================
#  PUBLIC API — Full pipeline for Flask /gradcam slider endpoint
# ================================================================
def regenerate_gradcam(
    image_path:      str,
    model,
    clinical_data:   np.ndarray,
    intensity:       float = GRADCAM_CONFIG["default_alpha"],
    last_conv_layer: str   = "auto",
) -> Optional[str]:
    """
    Full end-to-end Grad-CAM pipeline.
    Called by Flask /gradcam endpoint for live intensity slider.
    """
    from tensorflow.keras.preprocessing import image as kimage

    img       = kimage.load_img(image_path, target_size=(128, 128))
    img_array = kimage.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    heatmap = get_gradcam_heatmap(
        model, img_array, clinical_data, last_conv_layer
    )

    return superimpose_heatmap(image_path, heatmap, intensity=intensity)