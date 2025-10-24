from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app)

# Load the trained model and scaler
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Create label encoders with the same mapping used during training
gender_encoder = LabelEncoder()
gender_encoder.classes_ = np.array(['Female', 'Male'])

occupation_encoder = LabelEncoder()
occupation_encoder.classes_ = np.array(['Accountant', 'Doctor', 'Engineer', 'Lawyer', 
                                        'Manager', 'Nurse', 'Sales Representative', 
                                        'Salesperson', 'Scientist', 'Software Engineer', 'Teacher'])

bmi_encoder = LabelEncoder()
bmi_encoder.classes_ = np.array(['Normal', 'Normal Weight', 'Obese', 'Overweight'])

sleep_disorder_decoder = LabelEncoder()
sleep_disorder_decoder.classes_ = np.array(['Insomnia', 'None', 'Sleep Apnea'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Extract features from request
        gender = data['gender']
        age = float(data['age'])
        occupation = data['occupation']
        sleep_duration = float(data['sleepDuration'])
        quality_of_sleep = float(data['qualityOfSleep'])
        physical_activity = float(data['physicalActivity'])
        stress_level = float(data['stressLevel'])
        bmi_category = data['bmiCategory']
        heart_rate = float(data['heartRate'])
        daily_steps = float(data['dailySteps'])
        systolic = float(data['systolic'])
        diastolic = float(data['diastolic'])
        
        # Calculate Mean Blood Pressure
        mean_bp = (systolic + 2 * diastolic) / 3
        
        # Encode categorical variables
        gender_encoded = gender_encoder.transform([gender])[0]
        occupation_encoded = occupation_encoder.transform([occupation])[0]
        bmi_encoded = bmi_encoder.transform([bmi_category])[0]
        
        # Scale Mean_BP
        mean_bp_scaled = scaler.transform([[mean_bp]])[0][0]
        
        # Create feature array in the correct order
        # Order: Gender, Age, Occupation, Sleep Duration, Quality of Sleep, 
        #        Physical Activity Level, Stress Level, BMI Category, 
        #        Heart Rate, Daily Steps, Mean_BP
        features = np.array([[
            gender_encoded,
            age,
            occupation_encoded,
            sleep_duration,
            quality_of_sleep,
            physical_activity,
            stress_level,
            bmi_encoded,
            heart_rate,
            daily_steps,
            mean_bp_scaled
        ]])
        
        # Make prediction using YOUR trained model (no modifications)
        prediction_encoded = model.predict(features)[0]
        prediction_encoded = int(round(prediction_encoded))
        
        # Ensure prediction is within valid range (0, 1, 2)
        prediction_encoded = max(0, min(2, prediction_encoded))
        
        # Decode prediction back to text
        prediction = sleep_disorder_decoder.inverse_transform([prediction_encoded])[0]
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'mean_bp': round(mean_bp, 2)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': True})

def main():
    """Main entry point for the Flask application"""
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == '__main__':
    main()