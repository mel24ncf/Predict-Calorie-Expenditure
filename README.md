# Calorie Expenditure Prediction

Predicting calories burned during exercise using physiological and workout-based features from the Kaggle Playground Series S5E5 dataset. The final XGBoost regression pipeline is deployed as an interactive Streamlit web application.

👉 **[Try the live app](https://calorie-expenditure-prediction.streamlit.app/)** - enter workout and body metrics to generate a calorie prediction

**[Modeling notebook](notebooks/02_modeling.ipynb)** · **[EDA notebook](notebooks/01_load_clean_eda.ipynb)**

---

## Results

Four regression models were evaluated using RMSLE. XGBoost achieved the strongest performance, improving over linear and tree-based baselines while maintaining strong holdout generalization.

| Model            |   CV RMSLE |
| ---------------- | ---------: |
| Ridge Regression |     0.1504 |
| Decision Tree    |     0.0769 |
| Random Forest    |     0.0703 |
| **XGBoost** ✓    | **0.0621** |

After feature engineering and randomized hyperparameter tuning, the final model achieved:

| Metric                                |       Score |
| ------------------------------------- | ----------: |
| Kaggle Public RMSLE                   | **0.05924** |
| Mean CV-Test RMSLE — Baseline XGBoost |      0.0621 |
| Mean CV-Test RMSLE — Tuned XGBoost    |      0.0601 |
| Holdout Test RMSLE                    |      0.0602 |

Hyperparameter tuning reduced mean CV-test RMSLE from **0.0621 to 0.0601**, and the tuned model generalized closely to the holdout test set with **0.0602 RMSLE**.

---

## Why This Problem

Calorie expenditure is difficult to estimate accurately because it depends on both workout intensity and individual physiological characteristics. Duration alone is not enough: two people can exercise for the same amount of time but burn different calories depending on heart rate, body size, sex, body temperature, and intensity.

This project frames calorie prediction as a supervised regression problem and builds a deployable model that can estimate calories burned from user-provided workout and body metrics.

---

## Modeling Approach

### XGBoost as the champion model

XGBoost was selected as the final model because calorie expenditure is driven by nonlinear relationships between workout intensity and physiological features. Heart rate, duration, body temperature, weight, and engineered interaction terms can combine in ways that linear models may not capture well.

Compared with Ridge Regression, Decision Tree, and Random Forest baselines, XGBoost produced the lowest cross-validated RMSLE and strong holdout performance.

### Log-transforming the target

The final model is wrapped in a `TransformedTargetRegressor` using `log1p` and `expm1`. This aligns training with RMSLE, reduces the impact of large calorie values, and improves prediction stability for a positively skewed target.

### Validation strategy

Models were compared using cross-validated RMSLE. Hyperparameter tuning was performed with `RandomizedSearchCV` using shuffled K-fold cross-validation and a fixed `random_state` for reproducibility. Final performance was evaluated on a separate holdout test split.

---

## Feature Engineering

Three features were engineered to improve model signal:

| Feature               | Formula / Logic                           | Purpose                                        |
| --------------------- | ----------------------------------------- | ---------------------------------------------- |
| BMI                   | `weight / height²`                        | Captures body-size-adjusted mass               |
| BMR                   | Derived from sex, age, height, and weight | Estimates baseline energy expenditure          |
| Duration × Heart Rate | `Duration * Heart_Rate`                   | Captures combined workout length and intensity |

These were implemented in a custom sklearn-compatible transformer so feature generation occurs consistently during training and inference.

---

## Interpretability

SHAP analysis indicates that workout intensity features are the strongest drivers of predicted calorie expenditure. The engineered `Duration_X_HR` interaction term is the most influential feature, followed by physiological and workout variables such as heart rate, duration, body temperature, and weight.

<p align="center">
  <img src="artifacts/shap_summary_xgb.png" width="70%" />
</p>

---

## Pipeline Architecture

The final model is a fully reproducible sklearn pipeline, serialized as a `.joblib` artifact containing preprocessing, feature engineering, target transformation, and the trained XGBoost regressor:

```text
TransformedTargetRegressor
├── target transform: log1p / expm1
└── Pipeline
    ├── FeatureEngineering()          # BMI, BMR, Duration_X_HR
    ├── ColumnTransformer
    │   ├── Categorical: OneHotEncoder
    │   └── Numerical: passthrough
    └── XGBRegressor
```

This ensures that the same transformations used during training are applied during inference in the Streamlit app.

---

## Live Demo

Deployed with Streamlit Community Cloud · **[Open app](https://calorie-expenditure-prediction.streamlit.app/)**

The app:

* Accepts height in feet/inches
* Accepts weight in pounds
* Converts user inputs to metric units internally
* Builds a single-row prediction DataFrame
* Passes the input through the full trained pipeline
* Returns the predicted calories burned

---

## Repository Structure

```text
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
│   └── xgb_calories_model.joblib   # Trained XGBoost pipeline
├── notebooks/
│   ├── 01_load_clean_eda.ipynb     # Data loading, cleaning, and EDA
│   └── 02_modeling.ipynb           # Model training, tuning, and evaluation
├── src/
│   ├── __init__.py
│   └── feature_engineering.py      # Custom feature engineering transformer
├── requirements.txt                # Production dependencies
├── requirements-dev.txt            # Development dependencies
└── README.md
```

---

## Getting Started

```bash
# Clone and enter the repo
git clone https://github.com/melvinadkins/calorie-burn-prediction.git
cd calorie-burn-prediction

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app locally
streamlit run app/app.py
```

---

## Key Features

* End-to-end sklearn pipeline with feature engineering and XGBoost regression
* Target transformation using `log1p` / `expm1` for RMSLE-aligned modeling
* Custom feature engineering transformer for BMI, BMR, and workout intensity interaction
* Streamlit deployment with cached model loading
* Reproducible EDA and modeling notebooks
* Kaggle submission workflow with holdout validation

---

*Built using the Kaggle Playground Series S5E5 Predict Calorie Expenditure dataset.*
