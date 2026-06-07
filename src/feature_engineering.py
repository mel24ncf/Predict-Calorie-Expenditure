# Import libraries
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
 

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.required_columns = [
            "Sex", "Age", "Height", "Weight", "Duration", "Heart_Rate"
        ]

    def fit(self, X, y=None):
        return self 
    
    def transform(self, X):
        X = X.copy()

        # BMI 
        X["BMI"] = X["Weight"] / ((X["Height"] / 100) ** 2)

        # BMR (Mifflin-St Jeor)
        X["BMR"] = np.where(
            X["Sex"] == "male",
            10 * X["Weight"] + 6.25 * X["Height"] - 5 * X["Age"] + 5,
            10 * X["Weight"] + 6.25 * X["Height"] - 5 * X["Age"] - 161
        )

        # Interaction of workout intensity
        X["Duration_X_HR"] = X["Duration"] * X["Heart_Rate"]

        return X
