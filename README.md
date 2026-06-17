# Calorie Expenditure Prediction

A production-style end-to-end machine learning pipeline predicting calories burned during exercise using a tuned XGBoost Regressor, deployed as an interactive 
Streamlit web application.
 
[![Live App](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?logo=streamlit)](https://calorie-expenditure-prediction.streamlit.app/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle)](https://kaggle.com/competitions/playground-series-s5e5)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)

--- 

## 🔍 Project Overview

> Reade, W., & Park, E. (2025). *Predict Calorie Expenditure*. Kaggle.  
> https://kaggle.com/competitions/playground-series-s5e5

This project builds an end-to-end regression pipeline to predict calories burned during a workout session using physiological and workout based features:
* Age 
* Height
* Weight
* Session Duration
* Heart Rate
* Body Temperature
* Sex 

Four models were evaluated:
* Ridge Regression
* Decision Tree
* Random Forest
* XGBoost (Champion Model)

After hyperparameter tuning and feature engineering, the final model achieved:

🏆 Kaggle Public RMSLE: 0.05924

The trained model is wrapped in a TransformedTargetRegressor (log-transform on target) and deployed with Streamlit.

## Production Pipeline Architecture
The final model is a fully reproducible sklearn pipeline:

```
TransformedTargetRegressor (log1p / expm1)
└── Pipeline
    ├── FeatureEngineering()          # BMI, BMR, Duration_X_HR
    ├── ColumnTransformer
    │   ├── OneHotEncoder             # Sex
    │   └── passthrough               # Age, Height, Weight, Duration, Heart_Rate, Body_Temp
    └── XGBRegressor
```

This helps with 
- Reducing data leakage
- Consistent transformations at inference
- Preprocessing
- Saving full model (.joblib artifact)

## 📁 Repository Structure
```
calorie-burn-prediction/
├── app/
│   └── app.py                      # Streamlit web app
├── assets/
│   └── gym.jpeg                    # Header image
├── data/
│   ├── original/
│   │   ├── train.csv               # Original Kaggle training data
│   │   └── test.csv                # Original Kaggle test data
│   ├── train_split.csv             # Local training split
│   ├── test_split.csv              # Local test split
│   └── submission.csv              # Kaggle submission file
├── model/
│   └── xgb_calories_model.joblib   # Trained XGBoost pipeline (preprocessing + model)
├── notebooks/
│   ├── 01_load_clean_eda.ipynb     # Data loading, cleaning, and EDA
│   └── 02_modeling.ipynb           # Model training, tuning, and evaluation
├── src/
│   ├── __init__.py
│   └── feature_engineering.py      # Custom feature engineering transformer
├── requirements.txt                # Production dependencies (Streamlit Cloud)
├── requirements-dev.txt            # Development dependencies (Jupyter, SHAP, etc.)
└── README.md
```

---

## 🧠 Model Performance

| Metric | Score |
|--------|-------|
| Kaggle Public RMSLE | 0.05924 |
| Mean CV-Test RMSLE (Baseline XGBoost) | 0.0621 |
| Mean CV-Test RMSLE (Tuned XGBoost) | 0.0601 |
| Holdout Test RMSLE | 0.0602 |

Hyperparameter tuning via RandomizedSearchCV (20 iterations) reduced CV-Test RMSLE from 0.0621 to 0.0601, with the tuned model generalizing to a holdout test set (RMSLE 0.0602).

---

## 💡 Methodology
1️⃣ Exploratory Data Analysis

- Distribution inspection
- Outlier detection
- Correlation analysis

2️⃣ Feature Engineering

Custom sklearn transformer:

src/feature_engineering.py

Engineered features include:

- Body Mass Index (BMI)
- Basal Metabolic Rate (BMR)
- Duration_X_HR

---

3️⃣ Model Selection

Models compared using cross-validated RMSLE:

| Model | CV RMSLE |
|-------|----------|
| Ridge | 0.1504 |
| Decision Tree | 0.0769 |
| Random Forest | 0.0703 |
| XGBoost | 0.0621 |

XGBoost was selected due to:

- Lowest validation RMSLE
- Relatively small overfit between cv-train and cv-test RMSLE
- Fast inference time
- Strong ability to model nonlinear relationships

---

4️⃣ Hyperparameter Tuning

RandomizedSearchCV with:

- KFold (shuffle=True, fixed random_state)
- 20 iterations
- Optimized for neg_root_mean_squared_log_error
- Final model retrained on full training data.

---

5️⃣ SHAP Feature Importance

<img width="789" height="580" alt="1b1c4833-8e8d-4192-95e1-2cc308407d17" src="https://github.com/user-attachments/assets/7a102acf-a21d-44d3-9e52-2c61e369752a" />

SHAP analysis indicates that heart rate and duration are among the most influential predictors of calorie expenditure, and confirms that the engineered interaction term 
of duration x heart rate is the most important of all features used. Physiological features (heart rate, body temperature) also contribute.

---

## 🌐 Live Demo

👉 **[predict-calorie-expenditure.streamlit.app](https://calorie-expenditure-prediction.streamlit.app/)

The app:

- Accepts height (feet/inches)
- Accepts weight (lbs)
- Converts to metric internally
- Builds a single-row DataFrame
- Passes through full pipeline
- Returns calorie prediction

---

## 🚀 Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/melvinadkins/calorie-burn-prediction.git
cd calorie-burn-prediction
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the app locally**
```bash
streamlit run app/app.py
```

---

## 📌 Key Features

- End-to-end ML pipeline with preprocessing, feature engineering, and XGBoost Regressor
- Modular utility function for reusable feature engineering
- Streamlit app with cached model loading for performance
- Reproducible notebooks with clear separation of EDA and modeling

---

*Built as part of the Kaggle Playground Series (Season 5, Episode 5)*
