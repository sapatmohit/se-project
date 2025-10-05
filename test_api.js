import fetch from 'node-fetch';

async function testPrediction() {
  try {
    // Test case 1: Normal operation
    console.log('Test case 1: Normal operation');
    let response = await fetch('http://localhost:3001/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        airTemperature: 298.15,
        processTemperature: 308.15,
        rotationalSpeed: 1500,
        torque: 40,
        toolWear: 0,
        type: 'L'
      }),
    });

    let result = await response.json();
    console.log('Prediction result:', result);
    
    // Test case 2: High tool wear
    console.log('\nTest case 2: High tool wear');
    response = await fetch('http://localhost:3001/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        airTemperature: 298.15,
        processTemperature: 308.15,
        rotationalSpeed: 1500,
        torque: 40,
        toolWear: 220,
        type: 'L'
      }),
    });

    result = await response.json();
    console.log('Prediction result:', result);
    
    // Test case 3: High torque and speed
    console.log('\nTest case 3: High torque and speed');
    response = await fetch('http://localhost:3001/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        airTemperature: 298.15,
        processTemperature: 308.15,
        rotationalSpeed: 2800,
        torque: 85,
        toolWear: 100,
        type: 'H'
      }),
    });

    result = await response.json();
    console.log('Prediction result:', result);
    
  } catch (error) {
    console.error('Error:', error);
  }
}

testPrediction();