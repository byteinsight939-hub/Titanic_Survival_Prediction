# 🚢 Titanic Survival Prediction + Interactive Dashboard

This project predicts whether a passenger survived the Titanic disaster using machine learning models like Logistic Regression, KNN, SVM, and Decision Trees. It includes an **interactive Streamlit dashboard** where users can upload the dataset, explore the data, compare model performance, and test survival predictions based on custom passenger inputs.

---

## 📂 Dataset Used

- **Source**: [Kaggle Titanic Dataset by YasserH](https://www.kaggle.com/datasets/yasserh/titanic-dataset)
- **File Required**: `train.csv`

---

## 🚀 How to Run the Dashboard

1. Install required libraries:
    ```bash
    pip install -r requirements.txt
    ```

2. Run the Streamlit app:
    ```bash
    streamlit run titanic_dashboard.py
    ```

3. Upload the `train.csv` file via the dashboard.

---

## 💡 Features

- Upload Titanic dataset and preview
- Automatic data cleaning (missing values, encoding)
- Visualize survival by gender
- Train and compare ML models
- Predict survival for a custom passenger

---

## 🧠 Machine Learning Models

- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Naive Bayes  
- Support Vector Machine (SVM)  
- Decision Tree  

Dashboard shows a bar chart of model accuracies and uses the best one for predictions.

---

## 🎯 Interactive Prediction

User can input:
- Passenger Class
- Sex
- Age
- Siblings/Spouses
- Parents/Children
- Fare
- Embarkation Point

and get instant survival prediction.

---

## 📦 Requirements

```txt
pandas
numpy
scikit-learn
seaborn
matplotlib
streamlit
