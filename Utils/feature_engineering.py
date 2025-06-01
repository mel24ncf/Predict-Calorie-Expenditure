from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

import sys
import os

# Add the parent directory of the current file (project root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        # You can later add flags here to control which features to add
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()

        # === Validate Required Columns ===
        required_cols = {"Sex", "Weight", "Height", "Age", "Duration", "Heart_Rate", "Body_Temp"}
        missing = required_cols - set(X_.columns)
        if missing:
            raise ValueError(f"Missing columns for feature engineering: {missing}")

        # === Vectorized BMR Calculation ===
        is_male = X_["Sex"].str.lower() == "male"
        X_["BMR"] = (
            is_male * (88.362 + 13.397 * X_["Weight"] + 4.799 * X_["Height"] - 5.677 * X_["Age"]) +
            (~is_male) * (447.593 + 9.247 * X_["Weight"] + 3.098 * X_["Height"] - 4.330 * X_["Age"])
        )

        # === Polynomial & Interaction Features ===
        X_["Duration Heart_Rate"] = X_["Duration"] * X_["Heart_Rate"]
        X_["Duration^2"] = X_["Duration"] ** 2
        X_["Duration Body_Temp"] = X_["Duration"] * X_["Body_Temp"]
        X_["Heart_Rate Body_Temp"] = X_["Heart_Rate"] * X_["Body_Temp"]
        X_["Heart_Rate^2"] = X_["Heart_Rate"] ** 2
        X_["Body_Temp^2"] = X_["Body_Temp"] ** 2
        X_["BMI"] = X_["Weight"] / (X_["Height"] / 100) ** 2
        

        # === Final Feature Set to Return ===
        engineered_features = [
            "BMR", "Duration Heart_Rate", "Duration^2", "Duration Body_Temp",
            "Heart_Rate Body_Temp", "Heart_Rate^2", "Body_Temp^2", "Age", "BMI"
        ]

        return X_[engineered_features]