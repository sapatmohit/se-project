// Utility to parse XGBoost model for client-side inference
import modelMetadata from '../../AI/model_metadata.json';

// Type definitions for XGBoost model structure
export interface XGBoostNode {
  nodeid: number;
  depth?: number;
  split?: string;
  split_condition?: number;
  yes?: number;
  no?: number;
  missing?: number;
  children?: XGBoostNode[];
  leaf?: number;
}

export interface XGBoostTree {
  gain?: number;
  cover?: number;
  children?: XGBoostNode[];
  leaf?: number;
}

export interface XGBoostModel {
  version: number[];
  learner: {
    gradient_booster: {
      model: {
        gbtree_model_param: {
          num_trees: string;
          size_leaf_vector: string;
        };
        trees: XGBoostTree[];
      };
    };
    objective: {
      name: string;
      softmax_multiclass_param: {
        num_class: string;
      };
    };
  };
}

// Load model metadata
const metadata = modelMetadata;

/**
 * Fetch and parse XGBoost booster data
 * @returns Promise that resolves to parsed XGBoost model
 */
export async function loadXGBoostModel(): Promise<XGBoostModel> {
  try {
    // In a real implementation, we would fetch the actual model file
    // For GitHub Pages, we would need to convert the model to a web-compatible format
    // For now, we'll return a simplified model structure
    
    // Simulate async loading
    await new Promise(resolve => setTimeout(resolve, 300));
    
    // Return a simplified model structure for demonstration
    return {
      version: [1, 7, 3],
      learner: {
        gradient_booster: {
          model: {
            gbtree_model_param: {
              num_trees: "100",
              size_leaf_vector: "0"
            },
            trees: generateSampleTrees()
          }
        },
        objective: {
          name: "multi:softprob",
          softmax_multiclass_param: {
            num_class: metadata.num_classes.toString()
          }
        }
      }
    } as XGBoostModel;
  } catch (error) {
    console.error('Failed to load XGBoost model:', error);
    throw new Error(`Model loading failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

/**
 * Generate sample trees for demonstration
 * @returns Array of sample trees
 */
function generateSampleTrees(): XGBoostTree[] {
  // In a real implementation, this would parse actual tree structures
  // For demonstration, we'll generate simple sample trees
  
  const trees: XGBoostTree[] = [];
  const numClasses = parseInt(metadata.num_classes.toString());
  
  // Generate one tree per class for demonstration
  for (let i = 0; i < numClasses * 10; i++) {
    trees.push({
      gain: 0.5,
      cover: 100,
      children: [
        {
          nodeid: 0,
          split: "f0",
          split_condition: 0.5,
          yes: 1,
          no: 2,
          missing: 1,
          children: [
            {
              nodeid: 1,
              yes: -1,
              no: -1,
              missing: -1,
              leaf: Math.random() * 0.1
            },
            {
              nodeid: 2,
              yes: -1,
              no: -1,
              missing: -1,
              leaf: Math.random() * 0.1
            }
          ]
        }
      ]
    });
  }
  
  return trees;
}

/**
 * Traverse a single XGBoost tree
 * @param tree - XGBoost tree to traverse
 * @param features - Input features
 * @returns Leaf value
 */
export function traverseTree(tree: XGBoostTree, features: number[]): number {
  // In a real implementation, we would traverse the actual tree structure
  // For now, we'll return a simplified result based on features
  
  if (!tree.children || tree.children.length === 0) {
    return tree.leaf || 0;
  }
  
  // Simple traversal simulation
  let sum = 0;
  for (let i = 0; i < Math.min(features.length, 5); i++) {
    sum += features[i] * 0.01;
  }
  return sum;
}

/**
 * Get prediction from XGBoost model
 * @param model - Parsed XGBoost model
 * @param features - Input features
 * @returns Class probabilities
 */
export function getXGBoostPrediction(model: XGBoostModel, features: number[]): number[] {
  const numClasses = parseInt(model.learner.objective.softmax_multiclass_param.num_class);
  const trees = model.learner.gradient_booster.model.trees;
  
  // In a real implementation, we would:
  // 1. Traverse all trees in the model
  // 2. Sum leaf values for each class
  // 3. Apply softmax transformation
  
  // For demonstration, we'll implement a more realistic approach
  const logits = new Array(numClasses).fill(0);
  
  // Distribute tree contributions among classes
  for (let i = 0; i < trees.length; i++) {
    const classIndex = i % numClasses;
    const contribution = traverseTree(trees[i], features);
    logits[classIndex] += contribution;
  }
  
  // Apply softmax
  const maxLogit = Math.max(...logits);
  const expLogits = logits.map(logit => Math.exp(logit - maxLogit));
  const sumExpLogits = expLogits.reduce((sum, val) => sum + val, 0);
  return expLogits.map(val => val / sumExpLogits);
}