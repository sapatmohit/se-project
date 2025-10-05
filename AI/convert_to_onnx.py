"""
Convert XGBoost model to ONNX format for client-side inference
"""
import numpy as np
import xgboost as xgb
import json
import pickle
import os
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
import onnxruntime as ort

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

# Convert to ONNX
print("\n" + "="*50)
print("Converting XGBoost model to ONNX format...")
print("="*50)

try:
    # Define the input type for ONNX conversion
    # The model expects 12 features
    num_features = len(model_info["features"])
    initial_type = [('float_input', FloatTensorType([None, num_features]))]
    
    # Convert the XGBoost model to ONNX using onnxmltools
    onnx_model = onnxmltools.convert_xgboost(
        model,
        initial_types=initial_type,
        target_opset=12
    )
    
    # Save the ONNX model
    onnx_path = os.path.join(script_dir, "xgboost_model.onnx")
    onnxmltools.utils.save_model(onnx_model, onnx_path)
    
    print(f"✓ ONNX model saved to: {onnx_path}")
    
    # Verify the ONNX model
    print("\n" + "="*50)
    print("Verifying ONNX model...")
    print("="*50)
    
    # Load and run ONNX model
    ort_session = ort.InferenceSession(onnx_path)
    
    # Get input/output names
    input_name = ort_session.get_inputs()[0].name
    output_names = [output.name for output in ort_session.get_outputs()]
    
    print(f"Input name: {input_name}")
    print(f"Output names: {output_names}")
    
    # Run inference with ONNX model
    onnx_result = ort_session.run(None, {input_name: sample_input})
    
    print(f"\nONNX model output (label): {onnx_result[0]}")
    print(f"ONNX model output (probabilities) shape: {onnx_result[1].shape}")
    print(f"ONNX model output (probabilities):\n{onnx_result[1]}")
    
    # Compare results
    print("\n" + "="*50)
    print("Comparison:")
    print("="*50)
    print(f"Original XGBoost: {original_result[0]}")
    print(f"ONNX Model:       {onnx_result[1][0]}")
    
    # Check if results are close
    if np.allclose(original_result[0], onnx_result[1][0], rtol=1e-4, atol=1e-4):
        print("\n✓ SUCCESS: ONNX model produces equivalent results!")
    else:
        print("\n⚠ WARNING: Results differ slightly (this is usually acceptable)")
    
    print("\n" + "="*50)
    print("Conversion completed successfully!")
    print("="*50)
    print(f"\nONNX model ready for browser use at: {onnx_path}")
    print("\nNext steps:")
    print("1. Copy the ONNX model to your Next.js public directory")
    print("2. Install onnxruntime-web: npm install onnxruntime-web")
    print("3. Update your client code to load and use the ONNX model")
    
except Exception as e:
    print(f"\n✗ ERROR during conversion: {str(e)}")
    print("\nFalling back to saving binary model...")
    binary_model_path = os.path.join(script_dir, "xgboost_model.bin")
    model.save_model(binary_model_path)
    print(f"Binary model saved to {binary_model_path}")