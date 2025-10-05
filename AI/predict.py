import sys
import json
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the trained model from the correct path
model_path = os.path.join(script_dir, "xgboost_multiclass_model.json")
model = xgb.XGBClassifier()
model.load_model(model_path)

# Load encoders
type_enc_path = os.path.join(script_dir, "type_encoder.pkl")
target_enc_path = os.path.join(script_dir, "target_encoder.pkl")

import pickle
with open(type_enc_path, 'rb') as f:
    type_enc = pickle.load(f)
    
with open(target_enc_path, 'rb') as f:
    target_enc = pickle.load(f)

def engineer_features(data):
    """Apply the same feature engineering as in training"""
    # Create a DataFrame with the input data
    df = pd.DataFrame([data])
    
    # Encode Type
    df['Type_enc'] = type_enc.transform(df['Type'])
    
    # Feature engineering (same as training)
    df['Temp_Diff'] = df['Process_temperature_K'] - df['Air_temperature_K']
    df['Stress_Index'] = df['Torque_Nm'] * df['Rotational_speed_rpm']
    df['Torque_Speed_Ratio'] = df['Torque_Nm'] / (df['Rotational_speed_rpm'] + 1)
    
    # For single predictions, we simplify these features
    df['Torque_roll_mean5'] = 0
    df['Torque_roll_std5'] = 0
    df['Wear_diff'] = 0
    
    # Select features in the same order as training
    features = [
        'Type_enc',
        'Air_temperature_K',
        'Process_temperature_K',
        'Rotational_speed_rpm',
        'Torque_Nm',
        'Tool_wear_min',
        'Temp_Diff',
        'Stress_Index',
        'Torque_Speed_Ratio',
        'Torque_roll_mean5',
        'Torque_roll_std5',
        'Wear_diff'
    ]
    
    return df[features]

def predict_failure(input_data):
    """Make a prediction using the trained model"""
    try:
        # Engineer features
        X = engineer_features(input_data)
        
        # Make prediction
        probabilities = model.predict_proba(X)
        prediction = model.predict(X)[0]
        confidence = np.max(probabilities)
        
        # Decode prediction and replace underscores with spaces
        failure_type = target_enc.inverse_transform([prediction])[0]
        failure_type = failure_type.replace('_', ' ')
        
        return {
            "failureType": failure_type,
            "confidence": float(confidence),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise Exception(f"Prediction error: {str(e)}")

if __name__ == "__main__":
    try:
        # Read input data from command line argument
        if len(sys.argv) < 2:
            raise Exception("No input data provided")
            
        input_json = sys.argv[1]
        input_data = json.loads(input_json)
        
        # Rename fields to match training data
        formatted_data = {
            'Air_temperature_K': input_data['airTemperature'],
            'Process_temperature_K': input_data['processTemperature'],
            'Rotational_speed_rpm': input_data['rotationalSpeed'],
            'Torque_Nm': input_data['torque'],
            'Tool_wear_min': input_data['toolWear'],
            'Type': input_data['type']
        }
        
        # Make prediction
        result = predict_failure(formatted_data)
        
        # Output result as JSON
        print(json.dumps(result))
        
    except Exception as e:
        error_result = {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
        print(json.dumps(error_result))
        sys.exit(1)