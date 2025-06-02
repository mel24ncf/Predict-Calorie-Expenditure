# 🏋️ Predict Calorie Expenditure
A machine learning project that predicts the number of calories burned during a workout using CatBoost Regressor and a Streamlit app.

## 🔍 Project Overview
### 📚 Citation
Reade, W., & Park, E. (2025). *Predict Calorie Expenditure*. Kaggle.  
Available at: [https://kaggle.com/competitions/playground-series-s5e5](https://kaggle.com/competitions/playground-series-s5e5)


In this project, I built a regression model using CatBoost to predict the number of calories burned during a workout. CatBoost, Decision Tree, Random Forest, and XGBoost were used for model selection. I applied hyperparameter tuning and feature engineering to the second best performing model, CatBoost, achieving an RMSLE score of 0.06302 on the submission data and an RMSLE of 0.0587 of my test set. CatBoost was selected because it supports GPU acceleration, enabling faster tuning and training. The final result is an interactive web app powered by Streamlit that provides project background and model prediction for user inputs.

📁 Files in this repo
| File Name       | Description |
|----------------|-------------|
| app.py         | Streamlit app file |
| test.csv       | Data for Kaggle submission |
| model_report.json | json file containing model hyperparameter details |
| model_pipeline.zip | zip folder containing trained CatBoost model with scaling and feature engineering pipeline |
| 01_load_clean_eda.ipynb | Jupyter notebook for data loading, cleaning and exploratory data analysis |
| 02_modeling.ipynb | Jupyter notebook for model training, hyperparameter tuning, and evaluation |
| feature_engineering.py | Python file for feature interactions |
| img.webp  | Image for notebooks and streamlit app |
| environment.yml | yaml file contain packages for environment setup |

🧠 Model Performance
The final model has a mean average error of 2, meaning that on average, the prediction is within 2 calories of the actual calories burned. Additionally, the learning curves indicate that the model generalizes well. Improvements to this model may include
further hyperparameter tuning and/or ensemble model for prediction.

🌐 Web App
You can interact with the model using the Streamlit app. Users can input information about their workout and predict the number of calories burned.

📌 Key Features
End-to-end pipeline: preprocessing + CatBoost Regressor

Caches and modular functions for performance

💻 Live Demo
👉 [Check out the App ](https://predict-calorie-expenditure.streamlit.app/)

🚀 Getting Started
Set up the environment
To install all necessary dependencies, run:

```bash
conda env create -f environment.yml
conda activate calorie-env
