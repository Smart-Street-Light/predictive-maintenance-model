# app.py

import datetime
import json  # Import json for parsing JSON strings
import logging  # For logging
# import os
import re

# import joblib
import requests  # Used only for fetching data from ThingSpeak
# import torch
# import torch.nn as nn
from flask import Flask, jsonify, request
from flask_cors import CORS
from groq import Groq  # Import the Groq library

from alert import alert

# Allow all CORS policy
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the same MaintenanceNN architecture as in training
# class MaintenanceNN(nn.Module):
#     def __init__(self, input_size, output_size, hidden_size=64):
#         super(MaintenanceNN, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.fc3 = nn.Linear(hidden_size, output_size)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.fc3(x)  # No sigmoid here; applied during prediction
#         return x

# # Initialize Flask app
app = Flask(__name__)
CORS(app)

# # Directory where models and scaler are saved
# MODEL_DIR = 'trained_models_1L_6'

# # Load the scaler
# scaler_path = os.path.join(MODEL_DIR, 'scaler.joblib')
# if not os.path.exists(scaler_path):
#     raise FileNotFoundError(f"Scaler not found at {scaler_path}")
# scaler = joblib.load(scaler_path)
# logger.info("Scaler loaded.")
# logger.info(f"Scaler's expected features: {scaler.feature_names_in_}")

# # Define the expected feature list after preprocessing (no one-hot encoding)
# EXPECTED_FEATURES = [
#     'latitude',
#     'longitude',
#     'temperature',
#     'humidity',
#     'rainfall',
#     'wind_speed',
#     'ambient_light',
#     'operational_hours',
#     'usage_cycles',
#     'energy_consumption',
#     'voltage_level',
#     'fault_logs',
#     'led_lifespan',
#     'sensor_status',
#     'firmware_version',
#     'previous_maintenance',
#     'last_maintenance_days',
#     'vandalism_incidents',
#     'proximity_infrastructure'
# ]

# logger.info(f"Expected features: {EXPECTED_FEATURES}")

# # Load all models
# models = {}
# maintenance_types = []
# for filename in os.listdir(MODEL_DIR):
#     if filename.endswith('_model.pth'):
#         maintenance_type = filename.replace('_model.pth', '')
#         maintenance_types.append(maintenance_type)
#         model_path = os.path.join(MODEL_DIR, filename)

#         input_size = len(EXPECTED_FEATURES)
#         output_size = 1  # Binary classification

#         model = MaintenanceNN(input_size=input_size, output_size=output_size)
#         try:
#             model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
#             model.eval()  # Set to evaluation mode
#             models[maintenance_type] = model
#             logger.info(f"Loaded model for '{maintenance_type}' from {model_path}")
#         except Exception as e:
#             logger.error(f"Failed to load model for '{maintenance_type}': {e}")

# Define the mapping dictionaries based on training (if needed)
# SENSOR_STATUS_MAPPING = {
#     'active': 0,
#     'error': 1
# }

# PREVIOUS_MAINTENANCE_MAPPING = {
#     'scheduled': 0,
#     'unscheduled': 1
# }

# Define Groq API details using environment variables for security
GROQ_API_KEY = "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"  # Fetch from environment variable
if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY is not set. Please set it as an environment variable.")

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# Define ThingSpeak API details
THINGSPEAK_API_URL = 'https://api.thingspeak.com/channels/2735640/feeds.json?api_key=GZ7644UDVF648SNT&results=1'

# Root endpoint
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Maintenance Prediction API. Use the /predict endpoint to make predictions."})

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Step 1: Fetch data from ThingSpeak API
        thingSpeak_response = requests.get(THINGSPEAK_API_URL, timeout=10)  # Added timeout for reliability

        if thingSpeak_response.status_code != 200:
            logger.error(f"ThingSpeak API Error {thingSpeak_response.status_code}: {thingSpeak_response.text}")
            return jsonify({'error': 'Failed to fetch data from ThingSpeak API.'}), 500

        thingSpeak_data = thingSpeak_response.json()
        logger.info(f"ThingSpeak data fetched: {thingSpeak_data}")

        # Step 2: Extract the latest feed data
        feeds = thingSpeak_data.get('feeds', [])
        if not feeds:
            logger.error("No feeds found in ThingSpeak response.")
            return jsonify({'error': 'No data available from ThingSpeak API.'}), 500

        latest_feed = feeds[0]  # Assuming the first feed is the latest

        # Extract relevant fields from the latest feed
        latitude = latest_feed.get('field1', 'N/A')
        longitude = latest_feed.get('field2', 'N/A')
        voltage_level = latest_feed.get('field3', 'N/A')
        current_level = latest_feed.get('field4', 'N/A')
        temperature = latest_feed.get('field5', 'N/A')
        humidity = latest_feed.get('field6', 'N/A')
        ambient_light = latest_feed.get('field7', 'N/A')
        # Add more fields if necessary

        # Step 3: Construct the prompt for Groq API
        prompt = {
            "role": "user",
            "content": f"""Act as an expert in predictive maintenance detection of smart street lights. Given the following data related to street light sensor data that is brought from the sensors, predict the maintenance type categories (may be one or more possible) and provide the reason for each category (except for no maintenance required). The available fields include voltage level (in volts), current level (in ampere), temperature (in degree Celsius), humidity, and ambient light. The possible maintenance types are: 'Power Issue,' 'Overheating,' 'Scheduled Maintenance,' and 'No Maintenance.' ('No Maintenance' indicates the light is working fine) Analyze the data provided and classify each entry into one of these categories based on the available metrics. Make sure the prediction is very accurate, i.e., it should be 100% accurate and no misprediction should be there. The accuracy is very important, so first in-depth analyze the values that I am providing of the sensor data.
        
        If certain data points are missing or unclear, make reasonable, generalized, and practical assumptions to interpret the data meaningfully, such as assuming typical environmental ranges or standard device operational conditions if not specified.
        
        Sensors Data:
        The following represents one data entry in JSON format. Determine the maintenance type required (if not required, give 'No Maintenance Required') and return the probability for each class in the format given at the end of the prompt.
        
        {thingSpeak_data}
        "current_date_time": "{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        Strictly follow this below JSON output format and refrain from providing any other text, only provide the response of JSON nothing else, else error would come while parsing the response:
        
        {{
          "predictions" : [{{"maintenance_class_name" : "name_of_predicted_maintenance", "reason": "reason"}}, {{"maintenance_class_name_2" : "name_of_predicted_maintenance_2", "reason": "reason"}}]
        }}
        In predictions, only those maintenance types should be there which are most accurate, and if "No Maintenance" is the type, then no other types should come.
        """
        }

        logger.info(f"Constructed prompt for Groq API: {prompt}")

        # Step 4: Make a request to Groq API using the Groq library
        try:
            # Assuming the Groq library has a chat.completions.create() method similar to OpenAI's
            groq_response = groq_client.chat.completions.create(
                messages=[prompt],
                model="llama-3.3-70b-versatile"  # Specify the correct model if needed
            )
            logger.info(f"Groq response: {groq_response}")
        except Exception as e:
            logger.error(f"Groq API Request Error: {e}")
            return jsonify({'error': 'Failed to send request to Groq API.'}), 500

        # Step 5: Handle Groq's response
        try:
            # Access the content using attribute access
            prediction_text = groq_response.choices[0].message.content
            logger.info(f"Raw prediction text from Groq: {prediction_text}")

            cleaned_text = re.sub(r"^```(?:json)?\n", "", prediction_text)
            cleaned_text = re.sub(r"\n```$", "", cleaned_text)
            
            # Parse the JSON string directly
            predictions_data = json.loads(cleaned_text)
            predictions_array = predictions_data.get('predictions', [])
            logger.info(f"Extracted predictions array: {predictions_array}")
            
            if predictions_array and (not "No Maintenance" in predictions_array[0].get("maintenance_class_name", "")):    
                id = thingSpeak_data["channel"]["id"]
                latitude = thingSpeak_data["channel"]["latitude"]
                longitude = thingSpeak_data["channel"]["longitude"]
                location = f"latitude = {latitude}, location = {longitude}"
                class_name = predictions_array[0].get("maintenance_class_name", "")
                print(class_name)
                alert("",f"Street Light - {id} - {class_name}",location, json.dumps(thingSpeak_data))

            # Return the 'predictions' array as the API response        
            return jsonify({'predictions': predictions_array}), 200

        except (AttributeError, IndexError, json.JSONDecodeError) as e:
            logger.error(f"Error parsing Groq response: {e}")
            return jsonify({'error': 'Invalid response format from Groq API.'}), 500

    except Exception as e:
        logger.error(f"Error during prediction: {e}")  # Log the error
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    # For production, consider using a production-ready server like Gunicorn
    app.run(host='0.0.0.0', port=5025, debug=True)
