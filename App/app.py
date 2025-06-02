# Import packages
import streamlit as st 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from PIL import Image
import zipfile
import sys
import os

# Define base paths
app_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(app_dir, ".."))
model_zip_path = os.path.join(project_root, "Model", "final_pipeline.zip")
extract_path = os.path.join(project_root, "Model", "unzipped_model")
model_file = os.path.join(extract_path, "final_pipeline.pkl")

# Add utility path
sys.path.append(os.path.join(project_root, "Utils"))

# Create Title and Subheader
st.title("How Many Calories Did You Burn?")
st.subheader("🔥 Predict calories burned from your workout")

# Display an image 
image = Image.open("../Utils/gym.jpeg")
st.image(image, caption="Calorie Burn Prediction")

# Load the dataset
@st.cache_data
def load_data():
    """
    Load the training dataset.
    
    Returns:
    - DataFrame containing the training data
    """
    return pd.read_csv("../Data/train_sample.csv").drop(columns=['id'])
df = load_data()

# Load the model (final pipeline)
@st.cache_resource
def load_model():
    """
    Load the pre-trained model.
    
    Returns:
    - Loaded model
    """
    if not os.path.exists(model_file):
        with zipfile.ZipFile(model_zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)
    return joblib.load(model_file)
model = load_model()

# Set defaults for model inputs (heart rate and body temperature)
@st.cache_data 
def get_filtered_defaults(sex, age, height_cm, weight_kg, duration):
    """
    Get filtered defaults for heart rate and body temperature based on user inputs.

    Args:
    -----
    sex (str): Sex of the user
    age (int): Age of the user in years
    height_cm (float): Height of the user in centimeters
    weight_kg (float): Weight of the user in kilograms
    duration (int): Length of time in minutes for the workout

    Returns:
    --------
    - Tuple containing mean heart rate and body temperature
    """
    # Filter the dataset based on user inputs
    filtered = df[
        (df["Sex"] == sex) &
        (df["Age"].between(age -2, age + 2)) &
        (df["Height"].between(height_cm - 1, height_cm + 1)) &
        (df["Weight"].between(weight_kg - 2.5, weight_kg + 2.5)) &
        (df["Duration"].between(duration - 2, duration + 2))
    ]
    
    if len(filtered) >= 30:
        hr = int(filtered["Heart_Rate"].mean())
        temp_c = round(filtered["Body_Temp"].mean(), 2)
    else:
        hr = int(df["Heart_Rate"].mean())
        temp_c = round(df["Body_Temp"].mean(), 2)
        
    temp_f = round((temp_c * 9/5) + 32, 2)  
    return hr, temp_f

# Define a function to predict calories burned
def predict_calories(workout_data):
    """
    Predict calories burned based on workout data.
    
    Parameters:
    - workout_data: DataFrame containing workout metrics
    
    Returns:
    - Predicted calories burned
    """
    return model.predict(workout_data)

# Set page configuration
with st.expander("🔍 Preview of the Dataset"):
    st.markdown("This dataset contains various workout metrics and the corresponding calories burned.")
    st.dataframe(df.head(100))

with st.expander("📊 View Feature Relationships"):
    st.write("Explore how each workout metric influences calories burned.")
    fig = sns.pairplot(df.sample(5000, random_state=42), diag_kind='kde')
    st.pyplot(fig)

with st.expander("🧠 How the Model Works"):
    st.markdown("- Uses a CatBoost regressor with engineered features like Basal Metabolic Rate, Body Mass Index, and other feature interactions.")
    st.markdown("- Trained on workout data and evaluated with RMSLE.")

with st.expander("📌 About the Data"):
    st.markdown("""
    This app uses data from the Kaggle competition  
    **[Predict Calorie Expenditure (Playground Series - Season 5, Episode 5)](https://kaggle.com/competitions/playground-series-s5e5)**  
    by Walter Reade and Elizabeth Park.

    > Reade, W. & Park, E. (2025). *Predict Calorie Expenditure*. Kaggle.  
    > [https://kaggle.com/competitions/playground-series-s5e5](https://kaggle.com/competitions/playground-series-s5e5)
    """)

# Input features for prediction
with st.expander("Input Features"):
    st.markdown("Enter your workout metrics to predict calories burned.")

    # Sex 
    sex = st.selectbox("Sex", ["male", "female"])

    # Age 
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=30)

    # Height (feet and inches)
    hcol1, hcol2 = st.columns(2)
    with hcol1:
        feet = st.number_input("Height (feet)", min_value=1, max_value=8, value=5)
    with hcol2:
        inches = st.number_input("Height (inches)", min_value=0, max_value=11, value=7)
    
    height_cm = round((feet * 12 + inches) * 2.54, 1)  # Convert to cm

    # Weight (lbs)
    weight_lbs = st.number_input("Weight (lbs)", min_value=44, max_value=440, value=154)
    weight_kg = round(weight_lbs * 0.453592, 2) # Convert to kg

    # Duration
    duration = st.number_input("Duration (minutes)", min_value=3, max_value=30, value=10)

    # Smart Filtering for Mean Heart Rate and Body Temperature
    ## Based on input values above, we can set reasonable defaults
    hr_default, temp_f_default = get_filtered_defaults(sex, age, height_cm, weight_kg, duration)

    # Heart Rate
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=60, max_value=125, value=int(hr_default))

    # Body Temperature
    body_temp = st.number_input("Body Temperature (°F)", min_value=100.0, max_value=106.0, value=float(temp_f_default))
    body_temp_c = round((body_temp - 32) * 5/9, 2)  # Convert back to Celsius for prediction

    # Predict Calories Burned
    workout_data = pd.DataFrame({
        "Sex": [sex],
        "Age": [age],
        "Height": [height_cm],
        "Weight": [weight_kg],
        "Duration": [duration],
        "Heart_Rate": [heart_rate],
        "Body_Temp": [body_temp_c]
    })
    
    # Predict calories burned
    if st.button("Predict Calories Burned"):
        prediction = predict_calories(workout_data)
        st.success(f"Estimated Calories Burned: {round(prediction[0], 0)}")
                                 
def convert_df_to_csv(df):
    return df.to_csv(index=False)

# Add a download button for the dataset
csv = convert_df_to_csv(df)
st.download_button(
    label="Download Train Sample (350k) Dataset",
    data=csv,
    file_name="calories_dataset.csv",
    mime="text/csv",
)
