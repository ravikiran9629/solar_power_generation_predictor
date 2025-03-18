import streamlit as st
import pandas as pd
import pickle

# Load the transformations and the trained model
with open('yj_wd.pkl', 'rb') as f:
    yj_wd = pickle.load(f)

with open('yj_h.pkl', 'rb') as f:
    yj_h = pickle.load(f)
    
with open('scaler.pkl', 'rb') as f:
    scaler_transformer = pickle.load(f)

with open('solar_power_generation_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Streamlit app
st.title('Solar Power Generation Predictor')

# Create a form for user input
with st.form("solar_power_form"):
    distance_to_solar_noon = st.number_input('Distance to Solar Noon (rad)', min_value=0.00, max_value=1.5, format="%.4f")
    temperature = st.number_input('Temperature (°C) - Daily Average', min_value=0.0, max_value=100.0, format="%.1f")
    sky_cover = st.selectbox('Sky Cover', [0, 1, 2, 3, 4])
    wind_direction = st.number_input('Wind Direction (°) - Daily Average', min_value=0, max_value=360)
    wind_speed = st.number_input('Wind Speed (m/s)', min_value=0.0, max_value=60.0, format="%.2f")
    average_wind_speed_period = st.number_input('Average Wind Speed (m/s) - 3 Hour Measurement', min_value=0.0, max_value=60.0, format="%.2f")
    humidity = st.number_input('Humidity (%)', min_value=0, max_value=100)
    average_pressure_period = st.number_input('Average Pressure (inches of Hg) - 3 Hour Measurement', min_value=0.0, max_value=40.0, format="%.2f")

    # Submit button inside the form
    submit_button = st.form_submit_button("Predict")

# Only process the prediction if the form is submitted
if submit_button:
    data = {
        'sky_cover': [sky_cover],
        'distance_to_solar_noon': [distance_to_solar_noon],
        'temperature': [temperature],
        'wind_direction': [wind_direction],
        'wind_speed': [wind_speed],
        'average_wind_speed_period': [average_wind_speed_period],
        'humidity': [humidity],
        'average_pressure_period': [average_pressure_period],
    }
    
    input_data = pd.DataFrame(data, index=[0])
    
    # Apply Yeo-Johnson transformation on 'wind_direction_yj', 'humidity_yj'
    input_data[['wind_direction_yj']] = yj_wd.transform(input_data[['wind_direction']])
    input_data[['humidity_yj']] = yj_h.transform(input_data[['humidity']])
    
    # List of columns to apply scaler (excluding 'sky_cover')
    scaled_features = ['distance_to_solar_noon', 'temperature', 'wind_speed',
           'average_wind_speed_period', 'average_pressure_period',
           'wind_direction_yj', 'humidity_yj']
    
    # Apply scaling
    input_data[scaled_features] = scaler_transformer.transform(input_data[scaled_features])
    
    input_data = input_data.drop(['wind_direction', 'humidity'], axis=1)
    
    # Predict 
    transformed_prediction = loaded_model.predict(input_data)  # Get transformed target prediction
    
    # Define the min and max based on your dataset
    min_value = 0   # Update with actual min from your dataset
    max_value = 37000  # Update with actual max from your dataset
    
    # Normalize the predicted value to the range [0,1]
    progress_value = (transformed_prediction[0] - min_value) / (max_value - min_value)
    progress_value = min(max(progress_value, 0.0), 1.0)  # Ensure it's within [0,1]
    
    # Show result
    st.subheader("Predicted Power Generated (J) - 3 Hour Measurement")
    st.markdown(f"<h2 style='color: red; font-size: 32px; font-weight: bold;'>{transformed_prediction[0]:.2f}</h2>", unsafe_allow_html=True)
    
    # Progress bar representation
    progress_bar = st.progress(progress_value)

