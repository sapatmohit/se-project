// Client-side utilities for machine learning inference
import { loadModelMetadata, type ModelMetadata } from './xgboost-parser';

// Type definitions
interface PredictionInput {
  airTemperature: number;
  processTemperature: number;
  rotationalSpeed: number;
  torque: number;
  toolWear: number;
  type: 'L' | 'M' | 'H';
}

interface FeatureEngineeredData {
  Type_enc: number;
  Air_temperature_K: number;
  Process_temperature_K: number;
  Rotational_speed_rpm: number;
  Torque_Nm: number;
  Tool_wear_min: number;
  Temp_Diff: number;
  Stress_Index: number;
  Torque_Speed_Ratio: number;
  Torque_roll_mean5: number;
  Torque_roll_std5: number;
  Wear_diff: number;
}

// Type encoding mapping (same as in Python)
const typeEncoding: Record<string, number> = {
  'L': 0,
  'M': 1,
  'H': 2
};

/**
 * Engineer features for the ML model (same logic as in Python)
 * @param input - Raw sensor data
 * @returns Feature-engineered data ready for model inference
 */
export function engineerFeatures(input: PredictionInput): number[] {
  // Apply the same feature engineering as in training
  const engineered: FeatureEngineeredData = {
    Type_enc: typeEncoding[input.type] || 0,
    Air_temperature_K: input.airTemperature,
    Process_temperature_K: input.processTemperature,
    Rotational_speed_rpm: input.rotationalSpeed,
    Torque_Nm: input.torque,
    Tool_wear_min: input.toolWear,
    Temp_Diff: input.processTemperature - input.airTemperature,
    Stress_Index: input.torque * input.rotationalSpeed,
    Torque_Speed_Ratio: input.torque / (input.rotationalSpeed + 1),
    Torque_roll_mean5: 0, // Simplified for single predictions
    Torque_roll_std5: 0,  // Simplified for single predictions
    Wear_diff: 0          // Simplified for single predictions
  };

  // Return features in the same order as training
  return [
    engineered.Type_enc,
    engineered.Air_temperature_K,
    engineered.Process_temperature_K,
    engineered.Rotational_speed_rpm,
    engineered.Torque_Nm,
    engineered.Tool_wear_min,
    engineered.Temp_Diff,
    engineered.Stress_Index,
    engineered.Torque_Speed_Ratio,
    engineered.Torque_roll_mean5,
    engineered.Torque_roll_std5,
    engineered.Wear_diff
  ];
}

/**
 * Get model metadata
 * @returns Promise that resolves to model metadata including features and classes
 */
export async function getModelMetadata(): Promise<ModelMetadata> {
  return await loadModelMetadata();
}

/**
 * Softmax function to convert logits to probabilities
 * @param logits - Raw model outputs
 * @returns Probabilities for each class
 */
export function softmax(logits: number[]): number[] {
  const maxLogit = Math.max(...logits);
  const expLogits = logits.map(logit => Math.exp(logit - maxLogit));
  const sumExpLogits = expLogits.reduce((sum, val) => sum + val, 0);
  return expLogits.map(val => val / sumExpLogits);
}

/**
 * Get the predicted class and confidence
 * @param probabilities - Class probabilities
 * @param metadata - Optional model metadata (will be loaded if not provided)
 * @returns Promise that resolves to predicted class and confidence
 */
export async function getPredictionResult(
  probabilities: number[], 
  metadata?: ModelMetadata
): Promise<{ failureType: string; confidence: number; }> {
  const meta = metadata || await loadModelMetadata();
  const classes = meta.classes;
  const maxIndex = probabilities.indexOf(Math.max(...probabilities));
  const confidence = probabilities[maxIndex];
  const failureType = classes[maxIndex];
  
  return { failureType, confidence };
}