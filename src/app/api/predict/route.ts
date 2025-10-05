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
  } catch (error) {
    console.error('Prediction error:', error);
    return Response.json(
      { error: (error as Error).message || 'Failed to make prediction' }, 
      { status: 500 }
    );
  }
}