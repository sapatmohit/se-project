// Client-side XGBoost inference implementation
import { engineerFeatures, getPredictionResult } from './ml-utils';
import { loadXGBoostModel, getXGBoostPrediction, type XGBoostModel } from './xgboost-parser';

// Type definitions
interface PredictionInput {
  airTemperature: number;
  processTemperature: number;
  rotationalSpeed: number;
  torque: number;
  toolWear: number;
  type: 'L' | 'M' | 'H';
}

interface PredictionResponse {
  failureType: string;
  confidence: number;
  timestamp: string;
}

// Global variable to store the loaded model
let loadedModel: XGBoostModel | null = null;

/**
 * Load the XGBoost model from the booster file
 * @returns Promise that resolves when the model is loaded
 */
async function loadModel(): Promise<void> {
  if (loadedModel) {
    return;
  }
  
  try {
    console.log('Loading XGBoost model...');
    loadedModel = await loadXGBoostModel();
    console.log('XGBoost model loaded successfully');
  } catch (error) {
    console.error('Failed to load XGBoost model:', error);
    throw new Error(`Model loading failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

/**
 * Make a prediction using the XGBoost model on the client side
 * @param input - Sensor data input
 * @returns Prediction result
 */
export async function predictFailure(input: PredictionInput): Promise<PredictionResponse> {
  try {
    // Load model if not already loaded
    await loadModel();
    
    // Engineer features (same as in Python)
    const features = engineerFeatures(input);
    
    // Use the actual model for prediction
    if (loadedModel) {
      const probabilities = getXGBoostPrediction(loadedModel, features);
      console.log('Raw probabilities from model:', probabilities);
      
      const result = await getPredictionResult(probabilities);
      console.log('Prediction result:', result);
      
      return {
        failureType: result.failureType,
        confidence: result.confidence,
        timestamp: new Date().toISOString()
      };
    } else {
      throw new Error('Model not loaded');
    }
  } catch (error) {
    throw new Error(`Prediction failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}