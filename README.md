# 💻 Laptop Price Prediction using Machine Learning

An end-to-end Machine Learning project that predicts laptop prices based on hardware specifications and performance-related features. The project uses Ridge Regression with hyperparameter tuning and provides a simple Streamlit interface for real-time predictions.

---

## 📌 Project Overview

Laptop prices depend on several factors such as processor generation, RAM capacity, brand value, and hardware specifications. This project aims to estimate laptop prices using machine learning techniques and evaluate how different features influence the final prediction.

The workflow covers:

* Data Cleaning
* Feature Engineering
* Exploratory Data Analysis (EDA)
* Categorical Encoding
* Model Training
* Hyperparameter Tuning
* Model Evaluation
* Streamlit Deployment

---

## 📂 Dataset

The dataset contains information about laptops and their specifications, including:

| Feature     | Description                 |
| ----------- | --------------------------- |
| Brand       | Laptop manufacturer         |
| Name        | Laptop model                |
| Spec Rating | Overall specification score |
| Processor   | Processor details           |
| CPU         | CPU configuration           |
| RAM         | Installed RAM               |
| RAM Type    | DDR4, DDR5, LPDDR5, etc.    |
| Price       | Target variable             |

Dataset Size:

* 893 Records
* No Missing Values

---

## ⚙️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Streamlit
* Joblib

---

## 🧠 Machine Learning Pipeline

### Data Preprocessing

* Removed unnecessary columns
* Converted RAM values into numerical format
* Handled categorical variables using One-Hot Encoding

### Model Training

The following approach was used:

* Train-Test Split
* Ridge Regression
* Hyperparameter Tuning using GridSearchCV
* Cross Validation

### Best Hyperparameter

```python
{'model__alpha': 1}
```

---

## 📊 Model Performance

### R² Score

```text
0.803
```

The model explains approximately **80.3% of the variance** in laptop prices.

### Mean Absolute Error

```text
₹16,380.29
```

On average, the predicted laptop price differs from the actual price by approximately ₹16,380.

---

## 📁 Project Structure

```text
Laptop-Price-Prediction-ML/
│
├── app.py
├── train.py
├── ml1.csv
├── laptop_price_model.pkl
├── requirements.txt
├── README.md
├── .gitignore
```

## 🔮 Future Improvements

* Compare Ridge Regression with Random Forest Regressor
* Add Explainable AI using SHAP
* Improve Feature Selection
* Deploy Publicly using Streamlit Cloud
* Build a React + FastAPI Production Version

mpactful technology projects.
