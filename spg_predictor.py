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

distance_to_solar_noon = st.number_input('Distance to Solar Noon(rad)')
temperature = st.number_input('Temperature(°C) - Daily Average')
sky_cover = st.selectbox('Sky Cover', [0, 1, 2, 3, 4])
wind_direction = st.number_input('Wind Direction(°) - Daily Average')
wind_speed = st.number_input('Wind Speed(m/s)')
average_wind_speed_period = st.number_input('Average Wind Speed(m/s) - 3 Hour Measurement')
humidity = st.number_input('Humidity(%)')
average_pressure_period = st.number_input('Average Pressure(inches of Hg) - 3 Hour Measurement')
    
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

input_data = pd.DataFrame(data,index=[0])

# Apply Yeo-Johnson transformation on 'wind_direction_yj', 'humidity_yj'
input_data[['wind_direction_yj']] = yj_wd.transform(input_data[['wind_direction']])
input_data[['humidity_yj']] = yj_h.transform(input_data[['humidity']])


# List of columns to apply scaler (excluding 'sky_cover')
scaled_features = ['distance_to_solar_noon', 'temperature', 'wind_speed',
       'average_wind_speed_period', 'average_pressure_period',
       'wind_direction_yj', 'humidity_yj']

# Apply scaling
input_data[scaled_features] = scaler_transformer.transform(input_data[scaled_features])

input_data = input_data.drop(['wind_direction','humidity'],axis=1)
# Predict 
transformed_prediction = loaded_model.predict(input_data)  # Get transformed target prediction

# Show result
if st.button("Show Result"):
    st.subheader("Predicted Power Generated(J) - 3 Hour Measurement")
    st.write(f"**{transformed_prediction[0][0]:.2f}**")
