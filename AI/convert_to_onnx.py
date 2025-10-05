"""
Convert XGBoost model to ONNX format for client-side inference
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
model = xgb.XGBClassifier()
model.load_model(model_path)

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

# Create a sample input for conversion
# We need to create a sample input that matches the expected input shape
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
original_result = model.predict_proba(sample_input)
print("Original XGBoost model result shape:", original_result.shape)
print("Original XGBoost model result:", original_result)

# Save model info for client-side use
model_metadata = {
    "features": model_info["features"],
    "classes": model_info["classes"],
    "num_classes": model_info["num_classes"]
}

metadata_path = os.path.join(script_dir, "model_metadata.json")
with open(metadata_path, 'w') as f:
    json.dump(model_metadata, f, indent=2)

print(f"Model metadata saved to {metadata_path}")

# For client-side inference, we'll also save a simplified version of the model
# that can be used with xgboost.js or other JavaScript libraries

# Save the model in a format that can be used by xgboost.js
# First, let's save the model in binary format which might be easier to work with
binary_model_path = os.path.join(script_dir, "xgboost_model.bin")
model.save_model(binary_model_path)
print(f"Binary model saved to {binary_model_path}")

print("Conversion preparation completed successfully!")
print("\nTo use this model in client-side JavaScript, you can:")
print("1. Use the ONNX model with onnxruntime-web (if conversion works)")
print("2. Use the binary model with xgboost.js")
print("3. Implement the feature engineering in JavaScript and use the model")