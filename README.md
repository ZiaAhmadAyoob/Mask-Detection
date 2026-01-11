# 😷 Face Mask Detection using YOLOv11

A computer vision–based application that detects whether a person is **wearing a face mask or not** from images. The project uses **YOLOv11** for object detection and is deployed as a **Streamlit web application** for real-time inference.

---

## 🔍 Problem Statement
Monitoring face mask compliance is important for public health safety, especially in crowded environments. Manual monitoring is inefficient and difficult to scale. This project automates face mask detection using deep learning.

---

## 🎯 Purpose
- Detect mask and no-mask cases automatically
- Support public health and safety systems
- Provide an easy-to-use web interface for detection

---

## 🧠 Model & Dataset
- **Model:** YOLOv11 (Ultralytics)
- **Task:** Object Detection
- **Classes:** Mask, No Mask
- **Dataset:** Custom / public face mask dataset
- **Annotation Format:** YOLO

---

## ⚙️ Tech Stack
- Python
- YOLOv11
- OpenCV
- NumPy
- Streamlit

---

## 🏗️ Project Workflow
1. Dataset collection and annotation
2. Model training using YOLOv11
3. Model evaluation using Precision, Recall, and mAP
4. Deployment using Streamlit Cloud

---

## 🖥️ Web Application
The Streamlit app allows users to:
- Upload images
- Detect face mask compliance
- Visualize bounding boxes and class labels

🔗 **Live App:** https://mask-detection-app.streamlit.app/

