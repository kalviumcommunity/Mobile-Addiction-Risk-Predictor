
# 📱 Mobile Addiction Risk Predictor

A Machine Learning based web application that predicts a user's **mobile addiction risk level** using smartphone usage patterns and daily habits.

---

## 📌 Project Description

Smartphones have become an essential part of daily life. However, excessive mobile usage can negatively affect mental health, sleep quality, concentration, productivity, and academic performance.

Many users are unaware of how addicted they are to their phones. This project helps identify addiction risk early by analyzing daily smartphone usage behavior through Machine Learning.

The system predicts whether the user falls under:

- 🟢 Low Risk  
- 🟡 Medium Risk  
- 🔴 High Risk  

---

## 🎯 Problem Statement

Mobile addiction is increasing rapidly among students and working professionals. Excessive screen time, social media overuse, gaming addiction, and poor sleep habits can lead to:

- Reduced productivity  
- Lack of concentration  
- Sleep disorders  
- Anxiety and stress  
- Poor academic performance  
- Social isolation  

There is a need for a smart system that can detect addiction risk and create awareness.

---

## 💡 Proposed Solution

This project uses Machine Learning algorithms to analyze user inputs such as:

- Daily screen time  
- Sleep hours  
- Social media usage  
- Gaming hours  
- Number of phone pickups  
- Study / work hours  

Based on these inputs, the trained model predicts the user's addiction risk level.

---

## 🛠️ Tech Stack

### Programming Language
- Python

### Machine Learning Libraries
- Scikit-learn
- Pandas
- NumPy

### Data Visualization
- Matplotlib
- Seaborn

### Frontend / Deployment
- Streamlit

### Model Saving
- Pickle

### Version Control
- GitHub

---

## 📂 Project Structure

```text
mobile-addiction-risk-predictor/
│── data/
│   └── mobile_usage_dataset.csv
│
│── models/
│   └── addiction_model.pkl
│
│── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
│
│── app.py
│── requirements.txt
│── README.md
````

---

## ⚙️ Machine Learning Workflow

1. Data Collection
2. Data Cleaning
3. Feature Selection
4. Train-Test Split
5. Model Training
6. Model Evaluation
7. Model Saving
8. Streamlit Deployment

---

## 📊 Input Features

* Screen Time (hours/day)
* Sleep Hours
* Social Media Usage
* Gaming Hours
* Phone Pickups per Day
* Study Hours
* Work Hours

---

## 🤖 Algorithms Used

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree Classifier
* Random Forest Classifier

---

## 📈 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

## 🖥️ How to Run the Project

### Clone Repository

```bash
git clone <your-repository-link>
cd mobile-addiction-risk-predictor
```

### Install Requirements

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## 🎯 Expected Output

The system predicts one of the following:

* Low Addiction Risk
* Medium Addiction Risk
* High Addiction Risk

---

## 🌍 Real World Impact

This project can help:

* Students improve study focus
* Working professionals improve productivity
* Users reduce unhealthy screen habits
* Raise awareness about digital addiction

---

## 🔮 Future Enhancements

* Weekly usage reports
* Personalized detox suggestions
* Screen time tracker integration
* Mobile app version
* Real-time analytics dashboard

---
