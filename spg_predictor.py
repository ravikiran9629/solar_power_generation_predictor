import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    transformed_prediction = loaded_model.predict(input_data)[0]  # Get target prediction

    # Show result
    st.subheader("Predicted Power Generated (J) - 3 Hour Measurement")
    #st.markdown(f"<h2 style='color: blue; font-size: 28px; font-weight: bold;'>{transformed_prediction:.2f}</h2>", unsafe_allow_html=True)

    # vizualization of predicted value in comparision of min and max from dataset
    # Define min and max power values
    min_power = 0  
    max_power = 37000  

    # Determine color based on predicted value
    if transformed_prediction < max_power * 0.33:
        bar_color = '#FF6961'
    elif transformed_prediction < max_power * 0.66:
        bar_color = '#FFD700'
    else:
        bar_color = '#77DD77'

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(7, 0.4))  # Wider for better appearance

    # Draw bar with rounded edges using a rectangle
    ax.barh([''], [transformed_prediction], color=bar_color, height=0.1, edgecolor="black", linewidth=0.8)

    # Set limits and styling
    ax.set_xlim(min_power, max_power)
    ax.set_xticks(np.linspace(min_power, max_power, 6))
    ax.set_xticklabels([f"{int(val):,}" for val in np.linspace(min_power, max_power, 6)], fontsize=10)
    ax.set_yticks([])
    ax.set_xlabel("Power Generated (J)", fontsize=11, fontweight='bold')

    # Title with prediction value
    plt.title(f"Predicted Power: {transformed_prediction:,.2f} J", fontsize=8, fontweight='bold', color=bar_color)
    
    # Remove borders for a modern look
    for spine in ax.spines.values():
       spine.set_visible(False)

    # Display in Streamlit
    st.pyplot(fig)


