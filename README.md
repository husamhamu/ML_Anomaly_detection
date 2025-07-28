# ⚙️ Anomaly Detection in Pneumatic Cylinder Production

This repository contains the solution for the **Data Challenge**, focusing on detecting anomalies in the manufacturing of pneumatic cylinder bottom parts using machine learning models.

---

## 📑 Table of Contents

* [🎯 Motivation and Goals](#-motivation-and-goals)
* [📊 Data and Feature Exploration](#-data-and-feature-exploration)
* [🛠️ Concept and Methodology](#%EF%B8%8F-concept-and-methodology)
* [🤖 Machine Learning Models](#-machine-learning-models)
* [📈 Model Evaluation](#-model-evaluation)
* [🏆 Results](#-results)
* [✅ Final Model and Applicability](#-final-model-and-applicability)
* [🚀 Future Improvements](#-future-improvements)
* [⚡ How to Run](#-how-to-run)

---

## 🎯 Motivation and Goals

* **Background:** CNC-milling process of pneumatic cylinders.
* **Goal:** Develop a machine learning model to classify bottom parts as:

  * **False:** Anomaly
  * **True:** No anomaly

This ensures quality control before further production steps and helps guarantee product functionality early on.
📷 

<img width="735" height="487" alt="image" src="https://github.com/user-attachments/assets/2880053f-b039-49c0-ac06-42f2a16b22e3" />

---

## 📊 Data and Feature Exploration

* Separation of **true** and **false** parts.
* Time series analysis across multiple sensors.
* Computation of statistical features: **mean, RMS, kurtosis, skewness**, etc.
* Generated **900 features per data point**.
  📷 
  <img width="1290" height="550" alt="image" src="https://github.com/user-attachments/assets/b1c70a8a-5d72-4a6d-ac64-bea61fef0003" />

---

## 🛠️ Concept and Methodology

The data pipeline consists of:

1. **Feature Extraction** ✨
2. **Feature Selection** 🔍 (correlation-based, reduced 900 → 161 features)
3. **Data Split** ✂️ (80% train / 20% validation, stratified)
4. **Class Imbalance Handling** ⚖️ (SMOTE)
5. **Feature Scaling** 📏 (StandardScaler)

📷
<img width="1266" height="427" alt="image" src="https://github.com/user-attachments/assets/34e47823-dcb5-4b2a-a945-514fe7ea703a" />


---

## 🤖 Machine Learning Models

We experimented with the following models:

* **MLP (Multilayer Perceptron)** 🧠
* **SVM (Support Vector Machine)** 📐
* **Random Forest (RF)** 🌲

Each model’s hyperparameters were optimized based on trial and error.

---

## 📈 Model Evaluation

* Models evaluated on **20% test set**.
* Primary metric: **F1-score**.
  📷
  <img width="1287" height="259" alt="image" src="https://github.com/user-attachments/assets/6d28efc9-231c-435a-a36f-da6ab9db4389" />


---


## ✅ Final Model and Applicability

* Uses **all data channels**.
* Fast training but requires **heavy preprocessing**.
* Models are **accurate but not fully reliable** for real-world deployment yet.
  📷 

  <img width="1183" height="500" alt="image" src="https://github.com/user-attachments/assets/1fe06444-5869-4f82-80b4-733c0134b503" />


---

## 🚀 Future Improvements

* Stronger **oversampling** of anomaly parts (SMOTE tuning).
* Collection of **more sensor data** to improve reliability.

---

## ⚡ How to Run

```bash
# Clone this repository
git clone https://github.com/<your-username>/<repo-name>.git

# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py

# Evaluate the model
python evaluate.py
```

---

