# Predict-Calorie-Expenditure


# 🏋️ Predict Calorie Expenditure
A machine learning project that predicts the number of calories burned during a workout using CatBoost Regressor and a Streamlit app.

##🔍 Project Overview
### 📚 Citation
Reade, W., & Park, E. (2025). *Predict Calorie Expenditure*. Kaggle.  
Available at: [https://kaggle.com/competitions/playground-series-s5e5](https://kaggle.com/competitions/playground-series-s5e5)


In this project, I built a regression model using CatBoost to predict the number of calories burned during a workout. CatBoost, Decision Tree, Random Forest, and XGBoost were used for model selection. I applied hyperparameter tuning and feature engineering to the second best performing model, CatBoost, achieving an RMSLE score of on my test set. CatBoost was selected because it can leverage the power of gpu. The final result is an interactive web app powered by Streamlit that provides project background and model prediction for user inputs.

📁 Files in this repo
File Name	Description
Iris.csv	Dataset containing iris species data
iris_classifier.ipynb	Jupyter notebook used for data exploration, model training, and evaluation
requirements.yaml	Conda environment file for reproducibility
app.py	The main Streamlit application
util.py	Holds reusable functions for generating model performance visualizations
iris_svm_model_details.json	json file containing model hyperparameter details for the best model
iris_svm_pipeline.joblib	Pre-trained SVM model with scaling and tuning
label_encoder.joblib	Label encoder used to transform species labels
app.py	The main Streamlit application
environment.yaml	Environment file for reproducibility
🧠 Model Performance
The final model achieved ~97% accuracy on both training and validation sets, indicating good generalization and no signs of overfitting. I used GridSearchCV to tune the C, gamma, and kernel parameters of the SVM.

🌐 Web App
You can interact with the model using the Streamlit app. Users can input measurements via sliders and instantly see the predicted species. A dynamic plot visualizes where your flower lands in PCA space with GMM density contours and an annotated prediction arrow.

📌 Key Features
End-to-end pipeline: preprocessing + SVM

PCA and GMM for visualizing density in a dimensionality reduced feature space

Live visualization with prediction annotation in Streamlit

Caches and modular functions for performance

💻 Live Demo
👉 Check out the Iris Species Classifier App

🚀 Getting Started
Set up the environment
To install all necessary dependencies, run:

conda env create -f environment.yml
conda activate iris-env
