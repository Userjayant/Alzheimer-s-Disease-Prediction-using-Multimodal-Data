<div align="center">

# 🧠 Alzheimer's Disease Prediction
### *Multimodal Deep Learning with Explainable AI*

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![Anna University](https://img.shields.io/badge/Anna%20University-B.Tech%20AI%20%26%20DS-blue?style=flat-square)](https://www.annauniv.edu/)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

> A final-year B.Tech AI & Data Science project that combines **MRI brain imaging** and **clinical patient data** to predict Alzheimer's Disease using a multimodal deep learning model — with **Grad-CAM explainability** to highlight the regions of the brain driving each prediction.

</div>

---

## 📌 Overview

Alzheimer's Disease (AD) is one of the most prevalent neurodegenerative disorders, yet early and accurate diagnosis remains a clinical challenge. This project addresses that gap by building an end-to-end prediction pipeline that fuses two distinct data modalities:

- **MRI Brain Scans** — processed through a Convolutional Neural Network (CNN) for spatial feature extraction
- **Clinical Patient Data** — demographics, cognitive scores (MMSE, CDR), and biomarkers

The combined model produces a stage classification output, while **Grad-CAM** (Gradient-weighted Class Activation Mapping) renders an interpretable heatmap overlay — identifying *which regions of the brain* contributed most to the prediction.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Multimodal Fusion** | Simultaneously processes MRI images and structured clinical data |
| **CNN Architecture** | Custom convolutional network for spatial feature extraction from brain scans |
| **Grad-CAM Explainability** | Visual heatmap overlays that highlight high-attention brain regions |
| **Web Interface** | Flask-based UI for MRI upload, clinical input, and result display |
| **Interactive Intensity Slider** | Adjust Grad-CAM overlay strength in real time |
| **Automated Layer Detection** | Recursively auto-selects the deepest Conv2D layer for gradient computation |

---

## 🛠️ Tech Stack

**Backend & Machine Learning**
- Python 3.8+
- TensorFlow / Keras
- Flask
- NumPy · Pandas · Scikit-learn

**Computer Vision**
- OpenCV — Grad-CAM rendering, CLAHE contrast enhancement, contour detection

**Frontend**
- HTML5 · CSS3 · JavaScript

---

## 🏗️ Project Architecture

```
Input
 ├── MRI Brain Image (.jpg / .png)
 └── Clinical Data (age, MMSE score, CDR, etc.)
        │
        ▼
 Preprocessing
 ├── Image  : Resize (128×128) → Normalize → Augment
 └── Clinical: Impute → Scale → Encode
        │
        ▼
 Multimodal CNN Model
 ├── CNN Branch    → Spatial feature maps from MRI
 └── Dense Branch  → Feature embedding from clinical data
        │
   [Concatenate & Classify]
        │
        ▼
 Prediction Output
 ├── Stage : Non-Demented / Very Mild / Mild / Moderate
 └── Confidence Score
        │
        ▼
 Grad-CAM Visualization
 └── JET heatmap overlay on original MRI (contour boundaries + colorbar)
```

---

## 📁 Folder Structure

```
alzheimer-prediction/
│
├── static/
│   ├── css/                  # Stylesheets
│   ├── js/                   # Frontend scripts
│   └── uploads/              # Uploaded MRI images & Grad-CAM outputs
│
├── templates/
│   └── index.html            # Main web interface
│
├── model/
│   └── alzheimer_model.h5    # ← Place downloaded model here (see below)
│
├── gradcam.py                # Grad-CAM heatmap generation & overlay
├── app.py                    # Flask application & routing
├── requirements.txt          # Python dependencies
└── README.md
```

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/Userjayant/alzheimer-prediction.git
cd alzheimer-prediction
```

**2. Create and activate a virtual environment** *(recommended)*
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download the pre-trained model** *(see section below)*

**5. Run the application**
```bash
python app.py
```

**6. Open in browser**
```
http://127.0.0.1:5000
```

---

## 📦 Model Download

The trained model file (`.h5`) exceeds GitHub's file size limit and is hosted on Google Drive.

**Download link:**
> 🔗 [alzheimer_model.h5 — Google Drive](https://drive.google.com/file/d/1VSlc7w7gMcN9USd8In_JvDk8LHFnqXf5/view)

**After downloading, place the file at:**
```
alzheimer-prediction/
└── model/
    └── alzheimer_model.h5   ← here
```

> ⚠️ The application will not start without the model file placed in the correct directory.

---

## 🚀 Usage

1. **Launch the app** — run `python app.py`
2. **Open the web interface** at `http://127.0.0.1:5000`
3. **Upload an MRI brain scan** (JPG or PNG)
4. **Enter clinical data** — age, MMSE score, CDR, and other patient details
5. **Submit** to receive:
   - Predicted Alzheimer's stage with a confidence score
   - Grad-CAM heatmap overlay highlighting the brain's key attention regions
6. **Use the intensity slider** to interactively adjust the heatmap overlay strength

---

## 📊 Results & Output

**Prediction**
- Multi-class classification: `Non-Demented` · `Very Mild` · `Mild` · `Moderate`
- Per-class confidence score displayed alongside the result

**Grad-CAM Heatmap**
- JET colormap overlay: blue → cyan → green → yellow → **red** (highest attention)
- White contour boundaries mark the top-35% activation zones
- CLAHE-enhanced MRI background ensures brain anatomy remains visible under the overlay
- Vertical colorbar legend (HI / MED / LO) for clinical reference

**Sample Output**

| Input MRI | Grad-CAM Overlay |
|:---:|:---:|
| ![Input MRI](static/uploads/sample_mri.jpg) | ![Grad-CAM Output](static/uploads/sample_gradcam.jpg) |
| *Original brain scan* | *Red/yellow = high-attention regions* |

> 📝 *To display your own screenshots here, save them as `static/uploads/sample_mri.jpg` and `static/uploads/sample_gradcam.jpg` — they will render automatically in this table.*

---

## 🔮 Future Enhancements

- [ ] Integration of longitudinal patient data for progression tracking
- [ ] Training on larger, more diverse datasets (ADNI, OASIS)
- [ ] 3D volumetric MRI analysis for improved spatial context
- [ ] Real-time cloud deployment (AWS / GCP / Azure)
- [ ] Mobile-responsive interface and REST API for third-party integration
- [ ] Transformer-based cross-modal attention mechanisms for fusion

---

## 👤 Author

<table>
  <tr>
    <td align="center">
      <b>Jayant TN</b><br/>
      B.Tech — Artificial Intelligence & Data Science<br/>
      C. Abdul Hakeem College of Engineering and Technology<br/>
      <i>(Affiliated to Anna University, Chennai)</i><br/>
      <br/>
      <a href="https://github.com/Userjayant">
        <img src="https://img.shields.io/badge/GitHub-Userjayant-181717?style=flat-square&logo=github" alt="GitHub"/>
      </a>
      &nbsp;
      <a href="https://www.linkedin.com/in/jayant-tn-72759b243/">
        <img src="https://img.shields.io/badge/LinkedIn-Jayant%20TN-0A66C2?style=flat-square&logo=linkedin" alt="LinkedIn"/>
      </a>
      &nbsp;
      <a href="mailto:jayanttn0407@gmail.com">
        <img src="https://img.shields.io/badge/Email-jayanttn0407@gmail.com-EA4335?style=flat-square&logo=gmail&logoColor=white" alt="Email"/>
      </a>
    </td>
  </tr>
</table>

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).  
© 2025 Jayant TN — Developed as a Final Year B.Tech Project at C. Abdul Hakeem College of Engineering and Technology.

---

<div align="center">

*If you found this project useful or insightful, consider giving it a ⭐ on GitHub — it helps others discover the work.*

</div>
