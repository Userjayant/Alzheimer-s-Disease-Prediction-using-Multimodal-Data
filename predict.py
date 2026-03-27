import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from gradcam import get_gradcam_heatmap, superimpose_heatmap
 
# -----------------------------------------------
# Load model once at startup
# -----------------------------------------------
model = load_model("alzheimer_multimodal_model.h5")
 
# -----------------------------------------------
# Class labels
# -----------------------------------------------
classes = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
 
 
def predict_alzheimer(image_path, clinical_data, gradcam_intensity=0.6):
    """
    Predict Alzheimer's stage from MRI + clinical data.
 
    Args:
        image_path        : path to uploaded MRI image
        clinical_data     : np.array shape (1, 3) -> [age, mmse, cdr]
        gradcam_intensity : float 0-1, heatmap blend strength
 
    Returns:
        dict with prediction results
    """
    try:
        # Load and preprocess image
        img = image.load_img(image_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
 
        # Run model prediction
        pred = model.predict([img_array, clinical_data])
 
        predicted_class = classes[np.argmax(pred)]
        confidence = float(np.max(pred))  # 0.0 - 1.0
 
        # Extract clinical inputs
        age  = float(clinical_data[0][0])
        mmse = float(clinical_data[0][1])
        cdr  = float(clinical_data[0][2])
 
        # Self-validation
        explanation_flag = "Consistent"
        if predicted_class == "NonDemented" and mmse < 24:
            explanation_flag = "Mismatch detected"
        elif predicted_class == "MildDemented" and mmse > 26:
            explanation_flag = "Mismatch detected"
 
        # Correction engine
        final_class = predicted_class
        if explanation_flag == "Mismatch detected":
            final_class = "Uncertain - Needs Clinical Review"
 
        # Trust score
        trust_score = confidence * 100
        if explanation_flag == "Mismatch detected":
            trust_score -= 30
        if mmse < 15:
            trust_score += 5
        trust_score = max(0.0, min(100.0, trust_score))
 
        # Risk level
        if "Demented" in final_class:
            risk = "High Risk"
        elif "Uncertain" in final_class:
            risk = "Moderate Risk"
        else:
            risk = "Low Risk"
 
        # Explanation text
        if explanation_flag == "Mismatch detected":
            explanation = "Mismatch between MRI pattern and clinical scores detected."
        else:
            explanation = "MRI and clinical data are consistent."
 
        # Grad-CAM heatmap (JET colormap)
        gradcam_path = None
        try:
            heatmap = get_gradcam_heatmap(model, img_array, clinical_data, "conv2d")
            gradcam_path = superimpose_heatmap(
                image_path,
                heatmap,
                intensity=gradcam_intensity
            )
        except Exception as gc_err:
            print("Grad-CAM error:", gc_err)
 
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
 
    except Exception as e:
        print("Prediction Error:", e)
        return {
            "class":       "Error",
            "confidence":  0,
            "labels":      classes,
            "values":      [0] * len(classes),
            "gradcam_url": None,
            "risk":        "Unknown",
            "explanation": str(e),
            "trust_score": 0,
            "validation":  "Error",
        }