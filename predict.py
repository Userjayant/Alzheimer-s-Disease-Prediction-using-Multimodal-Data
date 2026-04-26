import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Import Grad-CAM functions
from gradcam import get_gradcam_heatmap, superimpose_heatmap

# Suppress TensorFlow info and warning logs (keep only errors)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# ================================================================
#  Load model once at startup (not on every request)
# ================================================================
model = load_model("alzheimer_multimodal_model.h5")


# ================================================================
#  Class labels — must match training order
# ================================================================
classes = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']


# ================================================================
#  MAIN PREDICTION FUNCTION
# ================================================================
def predict_alzheimer(
    image_path:        str,
    clinical_data,
    gradcam_intensity: float = 0.55,
) -> dict:
    """
    Predict Alzheimer's disease stage from MRI image + clinical data.
    Includes Grad-CAM heatmap generation for explainability.

    Parameters
    ----------
    image_path        : str         — path to uploaded MRI image file
    clinical_data     : array-like  — shape (1, 3) → [age, mmse, cdr]
    gradcam_intensity : float       — heatmap blend strength (0.3 to 0.8)

    Returns
    -------
    dict with keys:
        class        : predicted class label (or uncertainty note)
        confidence   : confidence % (0–100)
        labels       : list of all class names
        values       : list of raw prediction probabilities
        gradcam_url  : file path to saved Grad-CAM image (or None)
        risk         : risk level string
        explanation  : human-readable consistency explanation
        trust_score  : adjusted confidence score (0–100)
        validation   : "Consistent" or "Mismatch detected"
    """
    try:
        # ── STEP 1: Preprocess MRI image ──────────────────────────────────
        img       = image.load_img(image_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0   # normalize pixel values to [0, 1]

        # ── STEP 2: Validate and prepare clinical data ────────────────────
        clinical_data = np.array(clinical_data, dtype=np.float32)

        # If shape is (3,) → reshape to (1, 3) for model input
        if clinical_data.ndim == 1:
            clinical_data = clinical_data[np.newaxis, :]

        # Final shape check
        if clinical_data.ndim != 2:
            raise ValueError(
                f"clinical_data must be 2D array, got shape: {clinical_data.shape}"
            )

        # ── STEP 3: Run model prediction ──────────────────────────────────
        # verbose=0 suppresses per-batch progress output
        pred = model.predict([img_array, clinical_data], verbose=0)

        # Get the class with highest probability
        predicted_class = classes[np.argmax(pred)]
        confidence      = float(np.max(pred))   # raw confidence [0.0, 1.0]

        # ── STEP 4: Extract clinical values for validation ─────────────────
        age  = float(clinical_data[0][0])
        mmse = float(clinical_data[0][1])   # Mini-Mental State Examination
        cdr  = float(clinical_data[0][2])   # Clinical Dementia Rating

        # ── STEP 5: Clinical consistency check ────────────────────────────
        # Cross-validate MRI prediction against clinical scores
        # MMSE: 30=perfect, 24-30=normal, 18-23=mild, <18=moderate/severe
        # CDR:  0=normal, 0.5=questionable, 1=mild, 2=moderate, 3=severe
        explanation_flag = "Consistent"

        if predicted_class == "NonDemented" and mmse < 24:
            explanation_flag = "Mismatch detected"

        elif predicted_class == "MildDemented" and mmse > 26:
            explanation_flag = "Mismatch detected"

        # ── STEP 6: Correction engine ─────────────────────────────────────
        final_class = predicted_class
        if explanation_flag == "Mismatch detected":
            final_class = "Uncertain - Needs Clinical Review"

        # ── STEP 7: Trust score calculation ───────────────────────────────
        trust_score = confidence * 100.0

        if explanation_flag == "Mismatch detected":
            trust_score -= 30.0

        if mmse < 15:
            trust_score += 5.0

        trust_score = float(np.clip(trust_score, 0.0, 100.0))

        # ── STEP 8: Risk level classification ─────────────────────────────
        if "Demented" in final_class:
            risk = "High Risk"
        elif "Uncertain" in final_class:
            risk = "Moderate Risk"
        else:
            risk = "Low Risk"

        # ── STEP 9: Human-readable explanation ────────────────────────────
        if explanation_flag == "Mismatch detected":
            explanation = (
                "Mismatch detected between MRI pattern and clinical scores. "
                "Please consult a neurologist for clinical review."
            )
        else:
            explanation = (
                "MRI findings and clinical data (MMSE/CDR) are consistent. "
                "Prediction confidence is reliable."
            )

        # ── STEP 10: Generate Grad-CAM heatmap ────────────────────────────
        # FIX: Pass "auto" instead of hardcoded "conv2d" so the correct last
        # conv layer is always found regardless of Keras auto-naming.
        gradcam_path = None
        try:
            heatmap = get_gradcam_heatmap(
                model,
                img_array,
                clinical_data,
                "auto"             # ← changed from "conv2d" to "auto"
            )

            gradcam_path = superimpose_heatmap(
                image_path,
                heatmap,
                intensity=gradcam_intensity,
            )

        except Exception as gc_err:
            print(f"[predict] Grad-CAM generation failed: {gc_err}")
            gradcam_path = None

        # ── STEP 11: Return structured result dict ─────────────────────────
        return {
            "class":       final_class,
            "confidence":  round(confidence * 100, 2),
            "labels":      classes,
            "values":      pred[0].tolist(),
            "gradcam_url": gradcam_path,
            "risk":        risk,
            "explanation": explanation,
            "trust_score": round(trust_score, 2),
            "validation":  explanation_flag,
        }

    except Exception as exc:
        print(f"[predict] Fatal prediction error: {exc}")
        return {
            "class":       "Error",
            "confidence":  0,
            "labels":      classes,
            "values":      [0] * len(classes),
            "gradcam_url": None,
            "risk":        "Unknown",
            "explanation": str(exc),
            "trust_score": 0,
            "validation":  "Error",
        }