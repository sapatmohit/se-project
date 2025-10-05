import { spawn } from 'child_process';
import { NextRequest } from 'next/server';
import * as path from 'path';

// Types for our input data
interface PredictionInput {
  airTemperature: number;
  processTemperature: number;
  rotationalSpeed: number;
  torque: number;
  toolWear: number;
  type: 'L' | 'M' | 'H';
}

// Types for our prediction response
interface PredictionResponse {
  failureType: string;
  confidence: number;
  timestamp: string;
}

// Function to call Python script for prediction
function callPythonPrediction(input: PredictionInput): Promise<PredictionResponse> {
  return new Promise((resolve, reject) => {
    const pythonScriptPath = path.join(process.cwd(), 'AI', 'predict.py');
    
    // Prepare input data as JSON string
    const inputData = JSON.stringify(input);
    
    // Spawn Python process
    const python = spawn('python', [pythonScriptPath, inputData]);
    
    let stdoutData = '';
    let stderrData = '';
    
    python.stdout.on('data', (data) => {
      stdoutData += data.toString();
    });
    
    python.stderr.on('data', (data) => {
      stderrData += data.toString();
    });
    
    python.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`Python script exited with code ${code}: ${stderrData}`));
        return;
      }
      
      try {
        const result = JSON.parse(stdoutData);
        resolve(result);
      } catch (error) {
        reject(new Error(`Failed to parse Python output: ${stdoutData}`));
      }
    });
    
    python.on('error', (error) => {
      reject(new Error(`Failed to start Python process: ${error.message}`));
    });
  });
}

export async function POST(request: NextRequest) {
  try {
    // Check if request has form data (file upload)
    const contentType = request.headers.get('content-type') || '';
    
    if (contentType.includes('multipart/form-data')) {
      // Handle file upload
      const formData = await request.formData();
      const file = formData.get('file') as File;
      
      if (!file) {
        return Response.json(
          { error: 'No file provided' }, 
          { status: 400 }
        );
      }
      
      // Convert file to text and parse JSON
      const fileContent = await file.text();
      let inputData: PredictionInput;
      
      try {
        inputData = JSON.parse(fileContent);
      } catch (error) {
        return Response.json(
          { error: 'Invalid JSON in file' }, 
          { status: 400 }
        );
      }
      
      // Validate input
      if (inputData.airTemperature === undefined || inputData.processTemperature === undefined || 
          inputData.rotationalSpeed === undefined || inputData.torque === undefined || 
          inputData.toolWear === undefined || !inputData.type) {
        return Response.json(
          { error: 'Missing required fields in JSON file' }, 
          { status: 400 }
        );
      }
      
      // Call Python script for prediction
      const result = await callPythonPrediction(inputData);
      return Response.json(result);
    } else {
      // Handle JSON body
      const body: PredictionInput = await request.json();
      
      // Validate input
      if (body.airTemperature === undefined || body.processTemperature === undefined || 
          body.rotationalSpeed === undefined || body.torque === undefined || 
          body.toolWear === undefined || !body.type) {
        return Response.json(
          { error: 'Missing required fields' }, 
          { status: 400 }
        );
      }
      
      // Call Python script for prediction
      const result = await callPythonPrediction(body);
      return Response.json(result);
    }
  } catch (error) {
    console.error('Prediction error:', error);
    return Response.json(
      { error: (error as Error).message || 'Failed to make prediction' }, 
      { status: 500 }
    );
  }
}

// Add OPTIONS method for CORS preflight requests
export async function OPTIONS() {
  return new Response(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    },
  });
}