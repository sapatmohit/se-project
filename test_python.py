import json
from AI.predict import predict_failure

# Test data
test_data = {
    'airTemperature': 298.15,
    'processTemperature': 308.15,
    'rotationalSpeed': 1500,
    'torque': 40,
    'toolWear': 0,
    'type': 'L'
}

# Format data to match training format
formatted_data = {
    'Air_temperature_K': test_data['airTemperature'],
    'Process_temperature_K': test_data['processTemperature'],
    'Rotational_speed_rpm': test_data['rotationalSpeed'],
    'Torque_Nm': test_data['torque'],
    'Tool_wear_min': test_data['toolWear'],
    'Type': test_data['type']
}

# Make prediction
result = predict_failure(formatted_data)
print(json.dumps(result, indent=2))