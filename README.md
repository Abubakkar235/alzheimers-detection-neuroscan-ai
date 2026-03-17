# 🧠 NeuroScan AI – Alzheimer’s Detection System

## 🚀 Overview
NeuroScan AI is an AI-powered healthcare system that detects and classifies Alzheimer’s disease stages from MRI scans using a **dual-model deep learning architecture (CNN + EfficientNet)**. The system improves prediction reliability through model comparison and provides interpretable insights for clinical decision support.

This project demonstrates **end-to-end AI system design**, integrating deep learning models, backend development, and real-world healthcare deployment.

---

## 🧩 Key Features
- Multi-class Alzheimer’s stage classification  
- Dual-model architecture (CNN + EfficientNet)  
- Confidence scoring and model agreement analysis  
- Automated PDF medical report generation  
- Role-based authentication (Doctor, Patient, Admin)  
- Real-time inference (5–15 seconds)  

---

## 🏗️ Tech Stack
- **Backend:** Python, Flask  
- **AI/ML:** TensorFlow, Keras, CNN, EfficientNet  
- **Database:** MySQL  
- **Frontend:** HTML, CSS, Bootstrap  
- **Security:** bcrypt  
- **Reporting:** ReportLab  

---

## 🧪 Model Performance
- CNN Accuracy: ~94%  
- EfficientNet Accuracy: ~92%  
- Model Agreement: ~87%  
- Processing Time: ~4–15 seconds  

---

## ⚙️ System Workflow
1. Upload MRI scan  
2. Image preprocessing (resize, normalization)  
3. Dual-model prediction (CNN + EfficientNet)  
4. Decision fusion and comparison  
5. Visualization and PDF report generation  

---

## 📸 System Architecture & Algorithms

### 🧠 System Architecture
![Architecture](screenshots/architecture.png)

### ⚙️ Algorithms Used
![Algorithms](screenshots/algorithms.png)

---

## 📦 Installation

```bash
git clone https://github.com/Abubakkar235/alzheimers-detection-neuroscan-ai
cd alzheimers-detection-neuroscan-ai
pip install -r requirements.txt
python app.py
