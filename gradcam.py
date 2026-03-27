import numpy as np
import cv2
import os
import tensorflow as tf
 
 
# -------------------------------
# 🔥 GRAD-CAM HEATMAP GENERATOR
# -------------------------------
def get_gradcam_heatmap(model, img_array, clinical_data, last_conv_layer_name="conv2d"):
    """
    Generate Grad-CAM heatmap using the last conv layer.
    Returns a normalized heatmap (0.0 - 1.0).
    """
    try:
        # Build gradient model
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[
                model.get_layer(last_conv_layer_name).output,
                model.output
            ]
        )
 
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(
                [img_array, clinical_data], training=False
            )
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
 
        # Gradients of class score w.r.t. conv output
        grads = tape.gradient(class_channel, conv_outputs)
 
        # Pool gradients over spatial dims
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
 
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
 
        # Normalize 0–1
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
 
    except Exception as e:
        print(f"[GradCAM] Heatmap generation error: {e}")
        # Return blank heatmap
        return np.zeros((8, 8), dtype=np.float32)
 
 
# -------------------------------
# 🎨 SUPERIMPOSE WITH JET COLORMAP
# -------------------------------
def superimpose_heatmap(image_path, heatmap, intensity=0.6, save_dir="uploads"):
    """
    Superimpose Grad-CAM heatmap on original MRI with COLORMAP_JET.
    
    Args:
        image_path  : path to original MRI image
        heatmap     : numpy heatmap (H x W), values 0–1
        intensity   : blend alpha (0.0 = only original, 1.0 = only heatmap)
        save_dir    : folder to save output
 
    Returns:
        Path to saved Grad-CAM image
    """
    try:
        # Load original image
        orig = cv2.imread(image_path)
        if orig is None:
            raise ValueError(f"Cannot read image: {image_path}")
 
        h, w = orig.shape[:2]
 
        # ✅ Resize heatmap to match image
        heatmap_resized = cv2.resize(heatmap, (w, h))
 
        # ✅ Scale to 0–255
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
 
        # ✅ Apply COLORMAP_JET → Blue=low, Yellow=medium, Red=HIGH
        colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
 
        # ✅ Blend with original
        alpha = float(intensity)
        alpha = max(0.0, min(1.0, alpha))  # clamp 0–1
 
        superimposed = cv2.addWeighted(colormap, alpha, orig, 1 - alpha, 0)
 
        # ✅ Add colorbar legend on the right side
        superimposed = add_colorbar(superimposed)
 
        # ✅ Save output
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        out_filename = f"gradcam_{base_name}_i{int(alpha*100)}.jpg"
        out_path = os.path.join(save_dir, out_filename)
 
        cv2.imwrite(out_path, superimposed)
        return out_path
 
    except Exception as e:
        print(f"[GradCAM] Superimpose error: {e}")
        return None
 
 
# -------------------------------
# 🌈 COLORBAR LEGEND
# -------------------------------
def add_colorbar(image, bar_width=30, padding=10):
    """
    Append a vertical JET colorbar on the right side of the image.
    Labels: LOW (blue) → MED (yellow) → HIGH (red)
    """
    h, w = image.shape[:2]
 
    # Create gradient bar (high to low, top = red, bottom = blue)
    bar = np.zeros((h, bar_width, 3), dtype=np.uint8)
    for row in range(h):
        # top = 255 (red in JET), bottom = 0 (blue in JET)
        val = int(255 * (1 - row / h))
        color = cv2.applyColorMap(np.array([[val]], dtype=np.uint8), cv2.COLORMAP_JET)[0][0]
        bar[row, :] = color
 
    # Add label text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(bar, "HI",  (2, 18),      font, 0.38, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(bar, "MED", (2, h // 2),  font, 0.32, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(bar, "LO",  (2, h - 8),   font, 0.38, (255,255,255), 1, cv2.LINE_AA)
 
    # Separator line
    sep = np.full((h, padding, 3), 20, dtype=np.uint8)
 
    # Concatenate: original | gap | bar
    result = np.concatenate([image, sep, bar], axis=1)
    return result
 
 
# -------------------------------
# 🎚️ REGENERATE WITH NEW INTENSITY
# -------------------------------
def regenerate_gradcam(image_path, model, clinical_data, intensity=0.6, last_conv_layer="conv2d"):
    """
    Full pipeline: compute heatmap + superimpose at given intensity.
    Used by the /gradcam endpoint for the live intensity slider.
    """
    from tensorflow.keras.preprocessing import image as kimage
 
    img = kimage.load_img(image_path, target_size=(128, 128))
    img_array = kimage.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
 
    heatmap = get_gradcam_heatmap(model, img_array, clinical_data, last_conv_layer)
    out_path = superimpose_heatmap(image_path, heatmap, intensity=intensity)
    return out_path
 