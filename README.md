# Calorie Burn Predictor 📊🔥

**By Dhruv Doshi, Aditya Vyas & Aditya Karanwal**

[Calorie Burn Predictor App](https://calorie-predictor-hk5vjwxepjnodgay8ip6hr.streamlit.app/)


## Project Overview

Calorie Burn Predictor is a student-built, end-to-end data science project that estimates the number of calories burned during a workout using an XGBoost regression model. We wrapped our model in a user-friendly Streamlit app so anyone can input their workout metrics and instantly get a personalized calorie estimate.


## Motivation

As fitness enthusiasts, we often asked ourselves:
> “How many calories did I *really* burn at the gym today?”

While commercial trackers give ballpark figures, we wanted a transparent, data-driven tool that explains the “why” behind each prediction. This project merges our passion for fitness with hands-on machine learning.


## Features

- **Custom Inputs:** Gender, age, height, weight, workout duration, average heart rate, and average body temperature.
- **Real-Time Predictions:** Instant calorie estimates on submission.
- **Interactive Visuals:** Live data previews and feature importance explanations.
- **Open-Source:** Fully accessible code and dataset links.


## Dataset

Our dataset consists of real workout sessions from volunteer participants. Each record includes:

| Feature             | Description                                   |
|---------------------|-----------------------------------------------|
| Gender              | Male / Female                                 |
| Age (years)         | Participant age                               |
| Height (cm)         | Participant height                            |
| Weight (kg)         | Participant weight                            |
| Duration (minutes)  | Length of workout                             |
| Average Heart Rate  | Mean bpm recorded via fitness tracker band   |
| Average Body Temp   | Mean °C recorded via fitness tracker band    |
| Calories Burned     | Ground-truth calories (label)                |

> **Note:** We capture average heart rate & body temperature from wearables like Fitbit or Apple Watch, ensuring continuous, context-aware inputs.


## Data Preprocessing

1. **Missing Value Handling:** Applied mean imputation for numerical gaps.
2. **Encoding:** Converted gender to binary (0/1).
3. **Scaling:** Standardized continuous features for uniformity.


## Model Development

- **Algorithm:** XGBoost Regression
- **Why XGBoost?**
  - Handles mixed feature types seamlessly.
  - Built-in regularization (L1 & L2) to reduce overfitting.
  - Fast, scalable, and offers clear feature-importance metrics.

- **Hyperparameter Tuning:** We optimized learning rate, max depth, and number of estimators via k-fold cross-validation.
- **Performance:** Achieved Mean Absolute Error (MAE) of **~1–3 calories** on the test set.


## App Deployment

Our Streamlit app provides an interactive front-end:

1. **Input Panel:** Sliders & radio buttons to capture all features.
2. **Live Preview:** Table summarizing the inputs.
3. **Prediction Button:** Runs the XGBoost model and displays the calorie estimate.
4. **Info Panels:** Explain feature impacts, model training steps, and evaluation metrics.



## Results & Key Insights

- **Intensity Matters:** Duration (r≈0.99), heart rate (r≈0.93), and body temperature (r≈0.84) strongly correlate with calorie burn.
- **Demographic Effects:** Age shows a mild positive correlation (r≈0.20); gender differences reflect metabolic variations.
- **Feature Engineering Wins:** Incorporating body temperature from wearables boosted our model’s accuracy.


## Team Members

| Name               | Roll Number |
| ------------------ | ----------- |
| Aditya Karanwal    | 23UCS514    |
| Aditya Vyas        | 23UCS519    |
| Dhruv Doshi        | 23UCS568    |
