<div align="center">

# ❤️ Heart Disease Prediction Web Application

### Machine Learning-Based Clinical Decision Support System

A web application that predicts the likelihood of heart disease using Machine Learning algorithms based on patient health parameters.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-blue)

</div>

---

# 📖 Overview

Heart disease is one of the leading causes of death worldwide. Early prediction can assist healthcare professionals in making informed decisions and encourage timely medical intervention.

This project is a **Machine Learning-powered Heart Disease Prediction Web Application** that analyzes patient clinical data and predicts the likelihood of heart disease through an intuitive web interface.

---

# ✨ Features

- ❤️ Heart Disease Prediction
- 📊 Interactive User Interface
- ⚡ Real-Time Predictions
- 🧠 Machine Learning Model
- 🌐 FastAPI REST API
- 📈 Easy-to-use Dashboard
- 🚀 Deployment Ready

---

# 🛠️ Technology Stack

### Programming Language

- Python

### Machine Learning

- Scikit-learn
- XGBoost
- Pandas
- NumPy

### Backend

- FastAPI
- Uvicorn

### Frontend

- HTML
- CSS
- JavaScript

### Model Serialization

- Joblib

---

# 📂 Project Structure

```text
Heart-Disease-Prediction/
│
├── backend/
│   ├── app.py
│   ├── predict.py
│   └── model_loader.py
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── models/
│   └── heart_model.pkl
│
├── data/
│
├── notebooks/
│
├── requirements.txt
├── README.md
├── .gitignore
└── Dockerfile
```

---

# 📊 Input Features

The model predicts heart disease using features such as:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise-Induced Angina
- ST Depression
- Slope
- Number of Major Vessels
- Thalassemia

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/shak117/heart-disease-prediction.git
```

Navigate to the project

```bash
cd heart-disease-prediction
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the environment

Windows

```bash
venv\Scripts\activate
```

Linux/macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the FastAPI server

```bash
uvicorn backend.app:app --reload
```

Open the application in your browser.

---

# 🤖 Machine Learning Workflow

```text
Patient Data
      │
      ▼
Data Preprocessing
      │
      ▼
Feature Scaling
      │
      ▼
Trained ML Model
      │
      ▼
Prediction
      │
      ▼
Result Display
```

---

# 📈 Model Pipeline

- Data Collection
- Data Cleaning
- Feature Engineering
- Model Training
- Model Evaluation
- Model Serialization
- API Integration
- Frontend Deployment

---

# 📷 Screenshots

Add screenshots of:

- Home Page
- Prediction Form
- Prediction Result
- API Documentation (Swagger)

---

# 🚀 Future Enhancements

- Patient Report Generation
- Explainable AI (SHAP/LIME)
- Docker Deployment
- Cloud Deployment (AWS EC2)
- Authentication
- Prediction History
- Database Integration

---

# 👨‍💻 Author

**Shashank Pawar**

B.Sc. Data Science

GitHub: https://github.com/shak117

---

# 📄 License

This project is intended for educational and research purposes. It is not a substitute for professional medical advice or diagnosis.
