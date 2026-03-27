# 🧠 Alzheimer’s Disease Prediction using Multimodal Data

An AI-powered system for early detection of Alzheimer’s Disease using **MRI brain scans** and **clinical data (Age, MMSE, CDR)**. This project integrates deep learning with explainability techniques to provide accurate and interpretable predictions.

---

## 📌 Overview

Alzheimer’s disease is a progressive neurological disorder that affects memory and cognitive function. Early detection is crucial for effective treatment and care.

This project uses a **multimodal approach**, combining:

* 🧠 MRI Image Analysis (CNN-based Deep Learning)
* 📊 Clinical Data (Age, MMSE, CDR)

to improve prediction accuracy and provide meaningful insights.

---

## 🚀 Features

* ✅ MRI-based Alzheimer’s classification
* ✅ Clinical data integration (Multimodal AI)
* ✅ Grad-CAM Brain Attention Map (Explainable AI)
* ✅ MRI Comparison (Old vs Current scan)
* ✅ Sensitivity Analysis (MMSE impact)
* ✅ AI Clinical Report generation
* ✅ Patient history tracking (Database)
* ✅ Modern interactive UI

---

## 🧠 Multimodal Concept

Unlike traditional models, this project combines:

* **Image Model (CNN)** → Extracts features from MRI scans
* **Clinical Inputs** → Age, MMSE, CDR

👉 This improves prediction reliability and mimics real-world clinical decision-making.

---

## 🖼️ Brain Attention Map (Grad-CAM)

The system generates a **Grad-CAM heatmap** to highlight:

* 🔵 Low importance regions
* 🟡 Medium importance regions
* 🔴 High importance regions

👉 Helps doctors understand **why the model predicted a result**.

---

## 📊 Prediction Classes

* NonDemented
* VeryMildDemented
* MildDemented
* ModerateDemented

---

## 📁 Project Structure

```
Alzheimer-Multimodal-AI/
│
├── app.py
├── predict.py
├── database.py
├── gradcam.py
├── train_model.py
├── clinical_data.csv
├── requirements.txt
├── README.md
│
├── templates/
│   └── index.html
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```
git clone https://github.com/your-username/Alzheimer-Multimodal-AI.git
cd Alzheimer-Multimodal-AI
```

---

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

### 3️⃣ Download Model File

Due to GitHub file size limits, the trained model is hosted externally:

👉 **Download Model (.h5):**
https://drive.google.com/file/d/1VSlc7w7gMcN9USd8In_JvDk8LHFnqXf5/view

👉 Place the file in the project root folder:

```
Alzheimer-Multimodal-AI/
```

---

### 4️⃣ Run the application

```
python app.py
```

👉 Open in browser:

```
http://127.0.0.1:5000
```

---

## 📈 Technologies Used

* Python
* Flask
* TensorFlow / Keras
* OpenCV
* NumPy
* HTML, CSS (Tailwind)
* SQLite

---

## 🧪 Dataset

MRI images categorized into:

* Mild Demented
* Moderate Demented
* Non Demented
* Very Mild Demented

---

## 🎯 Future Enhancements

* 🔬 3D MRI Analysis
* 📊 Advanced Explainability (SHAP / LIME)
* 🌐 Cloud Deployment
* 📱 Mobile App Integration

---

## 👨‍💻 Authors

* **Jayant TN (22906)**
* **Mohammed Aamir Iqbal PA (22914)**
* **Syed Naveeth SN (22927)**

🎓 Final Year B.Tech – Artificial Intelligence & Data Science

---

## ⚠️ Disclaimer

This project is intended for **educational and research purposes only**.
It is **not a substitute for professional medical diagnosis**.

---

## ⭐ Acknowledgment

We thank our faculty and institution for guidance and support in completing this project.

---

## 🌐 License

This project is open-source and available for academic use.
