"""
Extract XGBoost model data for client-side inference
"""
import numpy as np
import xgboost as xgb
import json
import pickle
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the trained model from the correct path
model_path = os.path.join(script_dir, "xgboost_multiclass_model.json")
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(model_path)

# Load encoders
type_enc_path = os.path.join(script_dir, "type_encoder.pkl")
target_enc_path = os.path.join(script_dir, "target_encoder.pkl")

with open(type_enc_path, 'rb') as f:
    type_enc = pickle.load(f)
    
with open(target_enc_path, 'rb') as f:
    target_enc = pickle.load(f)

# Load model info
model_info_path = os.path.join(script_dir, "model_info.json")
with open(model_info_path, 'r') as f:
    model_info = json.load(f)

print("Model features:", model_info["features"])
print("Model classes:", model_info["classes"])

# Create a sample input for testing
sample_input = np.array([[
    0,          # Type_enc (0 for 'L')
    298.15,     # Air_temperature_K
    308.15,     # Process_temperature_K
    1500,       # Rotational_speed_rpm
    40,         # Torque_Nm
    0,          # Tool_wear_min
    10,         # Temp_Diff
    60000,      # Stress_Index
    0.0267,     # Torque_Speed_Ratio
    0,          # Torque_roll_mean5
    0,          # Torque_roll_std5
    0           # Wear_diff
]], dtype=np.float32)

print("Sample input shape:", sample_input.shape)

# Test the original model
original_result = xgb_model.predict_proba(sample_input)
print("Original XGBoost model result shape:", original_result.shape)
print("Original XGBoost model result:", original_result)

# Save model parameters for client-side implementation
model_params = {
    "feature_names": model_info["features"],
    "class_names": [cls.replace('_', ' ') for cls in target_enc.classes_.tolist()],
    "num_classes": model_info["num_classes"]
}

# Save the booster as a JSON string
booster_json = xgb_model.get_booster().save_raw()
booster_path = os.path.join(script_dir, "xgboost_booster.json")
with open(booster_path, 'wb') as f:
    f.write(booster_json)

print(f"XGBoost booster saved to {booster_path}")

# Save model parameters
params_path = os.path.join(script_dir, "tf_model_params.json")
with open(params_path, 'w') as f:
    json.dump(model_params, f, indent=2)

print(f"Model parameters saved to {params_path}")

# Also save a sample input and output for testing client-side implementation
test_data = {
    "input": sample_input.tolist(),
    "output": original_result.tolist()
}

test_path = os.path.join(script_dir, "test_data.json")
with open(test_path, 'w') as f:
    json.dump(test_data, f, indent=2)

print(f"Test data saved to {test_path}")
print("Conversion preparation completed successfully!")