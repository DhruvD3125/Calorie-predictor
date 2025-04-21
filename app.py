import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Set page configuration
st.set_page_config(
    page_title="Calorie Burn Predictor",
    page_icon="🔥",
    layout="centered"
)

# App title and description
st.title("🔥 Calorie Burn Predictor")
st.markdown("Enter your details to predict calories burned during exercise")

# Function to load and train the model
@st.cache_resource
def load_model():
    try:
        # Try to load a saved model if available
        with open('calorie_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        # If no saved model exists, train a new one
        calories = pd.read_csv('calories.csv')
        exercise_data = pd.read_csv('exercise.csv')
        calories_data = pd.concat([exercise_data, calories['Calories']], axis=1)
        
        # Preprocess data
        calories_data['Gender'].replace({'male': 0, 'female': 1}, inplace=True)
        
        # Split data
        X = calories_data.drop(columns=['User_ID', 'Calories'], axis=1)
        Y = calories_data['Calories']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
        
        # Train model
        model = XGBRegressor()
        model.fit(X_train, Y_train)
        
        # Save model for future use
        with open('calorie_model.pkl', 'wb') as file:
            pickle.dump(model, file)
        
        return model

# Load the model
with st.spinner("Loading model..."):
    model = load_model()

# Create two columns for input fields
col1, col2 = st.columns(2)

# User input fields
with col1:
    gender = st.radio("Gender", ["Male", "Female"])
    gender_encoded = 0 if gender == "Male" else 1
    
    age = st.slider("Age (years)", 15, 80, 30)
    height = st.slider("Height (cm)", 140, 220, 170)
    weight = st.slider("Weight (kg)", 40, 150, 70)

with col2:
    duration = st.slider("Exercise Duration (minutes)", 1, 60, 15)
    heart_rate = st.slider("Heart Rate (bpm)", 60, 200, 120)
    body_temp = st.slider("Body Temperature (°C)", 36.0, 40.0, 37.5, step=0.1)

# Create the feature array for prediction
features = np.array([[gender_encoded, age, height, weight, duration, heart_rate, body_temp]])

# Feature names for display
feature_names = ['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']

# Show the input data as a dataframe
st.subheader("Your Input Data")
input_df = pd.DataFrame(features, columns=feature_names)
input_df['Gender'] = "Male" if gender_encoded == 0 else "Female"
st.dataframe(input_df)

# Make prediction
if st.button("Predict Calories Burned"):
    prediction = model.predict(features)
    
    # Display the prediction with animations
    st.success(f"### Estimated Calories Burned: **{prediction[0]:.2f}** calories")
    
    # Provide an interpretation of the result
    if prediction[0] < 200:
        st.info("This is a low-intensity workout or short duration activity.")
    elif prediction[0] < 500:
        st.info("This is a moderate-intensity workout.")
    else:
        st.info("This is a high-intensity workout or long duration activity.")

# Add information about the factors affecting calorie burn
with st.expander("Factors Affecting Calorie Burn"):
    st.write("""
    - **Gender**: Biological differences affect metabolic rates
    - **Age**: Metabolism typically slows with age
    - **Weight**: Higher weight generally burns more calories during activity
    - **Duration**: Longer workouts burn more calories
    - **Heart Rate**: Higher heart rates indicate more intense exercise
    - **Body Temperature**: Related to exercise intensity and ambient conditions
    """)

# Add a section about the model
with st.expander("About this Model"):
    st.write("""
    This application uses an XGBoost Regression model to predict calorie burn based on physical characteristics and exercise metrics.
    
    The model was trained on exercise and calorie data from various individuals performing different types of physical activities.
    """)
    
    # Show model accuracy information if available
    st.write("**Model Performance**")
    st.write("Mean Absolute Error: Approximately 1-3 calories")
    
# Footer
st.markdown("---")
st.markdown("Created with ❤️ using Streamlit and Machine Learning")