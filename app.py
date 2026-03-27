import os
from flask import Flask, render_template, request, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from database import init_db, insert_data, get_history
from predict import predict_alzheimer
from gradcam import get_gradcam_heatmap, superimpose_heatmap
 
app = Flask(__name__)
init_db()
 
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}
 
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
 
 
# -------------------------------
# ✅ FILE CHECK
# -------------------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
 
 
# -------------------------------
# 🧠 CLINICAL REPORT
# -------------------------------
def generate_clinical_report(pred_class, confidence, age, mmse, cdr):
    if pred_class == "NonDemented":
        stage = "No signs of cognitive decline"
        risk   = "Low Risk"
    elif pred_class == "VeryMildDemented":
        stage = "Very early stage"
        risk   = "Moderate Risk"
    elif pred_class == "MildDemented":
        stage = "Mild Alzheimer's"
        risk   = "High Risk"
    else:
        stage = "Severe Alzheimer's"
        risk   = "Critical Risk"
 
    if confidence > 90:
        reliability = "Very High"
    elif confidence > 75:
        reliability = "High"
    else:
        reliability = "Moderate"
 
    return f"""
🧠 AI Clinical Report
 
Prediction: {pred_class}
Confidence: {confidence:.2f}%
 
Patient:
Age: {age}
MMSE: {mmse}
CDR: {cdr}
 
Stage:
{stage}
 
Risk:
{risk}
 
Reliability:
{reliability}
 
Recommendation:
Consult neurologist.
"""
 
 
# -------------------------------
# 🏠 HOME
# -------------------------------
@app.route("/")
def index():
    return render_template("index.html")
 
 
# -------------------------------
# 🔍 PREDICT
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    print("FORM:", request.form)
    print("FILES:", request.files)
 
    # File check
    if "image" not in request.files:
        return render_template("index.html", error="No image uploaded")
 
    file = request.files["image"]
    old_file = request.files.get("old_image")
 
    if file.filename == "":
        return render_template("index.html", error="No file selected")
 
    if not allowed_file(file.filename):
        return render_template("index.html", error="Invalid file type")
 
    # Save image
    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)
 
    # Old image
    old_path     = None
    old_filename = None
    if old_file and old_file.filename != "":
        if allowed_file(old_file.filename):
            old_filename = secure_filename(old_file.filename)
            old_path     = os.path.join(app.config["UPLOAD_FOLDER"], old_filename)
            old_file.save(old_path)
 
    # Clinical inputs
    def to_float(name, default=0.0):
        try:
            return float(request.form.get(name, default))
        except:
            return default
 
    age  = to_float("age")
    mmse = to_float("mmse")
    cdr  = to_float("cdr")
 
    # Grad-CAM intensity from form (default 0.6)
    gradcam_intensity = to_float("gradcam_intensity", 0.6)
 
    clinical_data = np.array([[age, mmse, cdr]], dtype=np.float32)
 
    # Prediction
    prediction = predict_alzheimer(save_path, clinical_data, gradcam_intensity=gradcam_intensity)
    print("Prediction:", prediction)
 
    pred_class = prediction.get("class", "Unknown")
    confidence = float(prediction.get("confidence", 0.0))
 
    # Contribution analysis
    image_weight = 0.6
    mmse_weight  = 0.25
    cdr_weight   = 0.15
 
    if confidence < 70:
        image_weight = 0.5
        mmse_weight  = 0.3
        cdr_weight   = 0.2
 
    prediction["contributions"] = [
        {"name": "MRI Scan",    "value": image_weight * 100},
        {"name": "MMSE Score",  "value": mmse_weight  * 100},
        {"name": "CDR Rating",  "value": cdr_weight   * 100},
    ]
 
    # Risk + explanation
    if pred_class == "NonDemented":
        prediction["risk"] = "Low Risk"
    elif pred_class == "VeryMildDemented":
        prediction["risk"] = "Moderate Risk"
    elif pred_class == "MildDemented":
        prediction["risk"] = "High Risk"
    else:
        prediction["risk"] = "Critical Risk"
 
    prediction["explanation"] = "AI used MRI + clinical data for prediction."
 
    # Old MRI comparison
    if old_path:
        old_prediction = predict_alzheimer(old_path, clinical_data)
        old_class      = old_prediction.get("class", "Unknown")
        old_conf       = float(old_prediction.get("confidence", 0.0))
        progression    = "No change" if old_class == pred_class else f"{old_class} → {pred_class}"
 
        prediction["old_class"]   = old_class
        prediction["old_conf"]    = old_conf
        prediction["progression"] = progression
 
    # Database
    insert_data(age, mmse, cdr, pred_class, confidence)
 
    # Clinical report
    prediction["report"] = generate_clinical_report(pred_class, confidence, age, mmse, cdr)
 
    # Sensitivity analysis
    sensitivity = []
    for d in range(6):
        new_mmse = max(0, mmse - d)
        temp     = np.array([[age, new_mmse, cdr]], dtype=np.float32)
        temp_pred = predict_alzheimer(save_path, temp)
        sensitivity.append({
            "mmse":       new_mmse,
            "class":      temp_pred.get("class"),
            "confidence": float(temp_pred.get("confidence", 0))
        })
    prediction["sensitivity"] = sensitivity
 
    # Image URL
    prediction["image_url"] = url_for("uploaded_file", filename=filename)
    # Store original filename + clinical inputs for the intensity slider endpoint
    prediction["original_filename"] = filename
    prediction["age"]  = age
    prediction["mmse"] = mmse
    prediction["cdr"]  = cdr
 
    if old_filename:
        prediction["old_image_url"] = url_for("uploaded_file", filename=old_filename)
 
    # Grad-CAM URL
    if prediction.get("gradcam_url"):
        gradcam_filename         = os.path.basename(prediction["gradcam_url"])
        prediction["gradcam_url"] = url_for("uploaded_file", filename=gradcam_filename)
 
    # History
    prediction["history"] = get_history()
 
    return render_template("index.html", prediction=prediction)
 
 
# -------------------------------
# 🎚️ LIVE GRAD-CAM INTENSITY ENDPOINT
# Used by the intensity slider via fetch()
# -------------------------------
@app.route("/gradcam", methods=["POST"])
def gradcam_regenerate():
    """
    Regenerate Grad-CAM at a new intensity without re-running the full model.
    Expects JSON: { "filename": "...", "age": 0, "mmse": 0, "cdr": 0, "intensity": 0.6 }
    Returns JSON: { "gradcam_url": "..." }
    """
    try:
        data      = request.get_json()
        filename  = secure_filename(data.get("filename", ""))
        intensity = float(data.get("intensity", 0.6))
        age       = float(data.get("age",  0))
        mmse      = float(data.get("mmse", 0))
        cdr       = float(data.get("cdr",  0))
 
        image_path    = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        clinical_data = np.array([[age, mmse, cdr]], dtype=np.float32)
 
        if not os.path.exists(image_path):
            return jsonify({"error": "Image not found"}), 404
 
        # Re-run only Grad-CAM (no full model prediction)
        from tensorflow.keras.preprocessing import image as kimage
        from predict import model
 
        img       = kimage.load_img(image_path, target_size=(128, 128))
        img_array = kimage.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
 
        heatmap  = get_gradcam_heatmap(model, img_array, clinical_data, "conv2d")
        out_path = superimpose_heatmap(image_path, heatmap, intensity=intensity)
 
        if out_path is None:
            return jsonify({"error": "Grad-CAM failed"}), 500
 
        gradcam_filename = os.path.basename(out_path)
        gradcam_url      = url_for("uploaded_file", filename=gradcam_filename)
 
        return jsonify({"gradcam_url": gradcam_url})
 
    except Exception as e:
        print("GradCAM endpoint error:", e)
        return jsonify({"error": str(e)}), 500
 
 
# -------------------------------
# 📁 SERVE FILES
# -------------------------------
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)
 
 
# -------------------------------
# 🚀 RUN
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)