# alzheimers-detection-neuroscan-ai
# 🧠 NeuroScan AI – Alzheimer’s Detection System

## 🚀 Overview
NeuroScan AI is an AI-powered healthcare system that detects and classifies Alzheimer’s disease stages from MRI scans using a **dual-model deep learning architecture (CNN + EfficientNet)**. The system improves prediction reliability through model comparison and provides interpretable insights for clinical decision support.

This project demonstrates **end-to-end AI system design**, combining deep learning, backend development, and real-world healthcare application deployment.

---

## 🧩 Key Features
- 🧠 Multi-class Alzheimer’s stage classification  
- 🤖 Dual-model architecture (CNN + EfficientNet)  
- 📊 Confidence scoring & model agreement analysis  
- 📄 Automated PDF medical reports  
- 🔐 Role-based authentication (Doctor, Patient, Admin)  
- ⚡ Real-time inference (5–15 seconds)  

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
2. Image preprocessing  
3. Dual-model prediction  
4. Decision fusion & comparison  
5. Visualization and PDF report generation  

---

## 📦 Installation

```bash
git clone https://github.com/Abubakkar235/alzheimers-detection-neuroscan-ai
cd alzheimers-detection-neuroscan-ai
pip install -r requirements.txt
python app.py
