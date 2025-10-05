// Utility to parse XGBoost model for client-side inference

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
  base_weights?: number[];
  categories?: unknown[];
  categories_nodes?: unknown[];
  categories_segments?: unknown[];
  categories_sizes?: unknown[];
  default_left?: number[];
  id?: number;
  left_children?: number[];
  loss_changes?: number[];
  parents?: number[];
  right_children?: number[];
  split_conditions?: number[];
  split_indices?: number[];
  split_type?: number[];
  sum_hessian?: number[];
  tree_param?: {
    num_deleted: string;
    num_feature: string;
    num_nodes: string;
    size_leaf_vector: string;
  };
}

export interface XGBoostModel {
  learner: {
    attributes?: Record<string, string>;
    feature_names: string[];
    feature_types: string[];
    gradient_booster: {
      model: {
        gbtree_model_param: {
          num_parallel_tree: string;
          num_trees: string;
        };
        iteration_indptr: number[];
        tree_info: number[];
        trees: XGBoostTree[];
      };
      name: string;
    };
    learner_model_param: {
      base_score: string;
      boost_from_average: string;
      num_class: string;
      num_feature: string;
    };
    objective: {
      name: string;
      softmax_multiclass_param: {
        num_class: string;
      };
    };
  };
}

export interface ModelMetadata {
  features: string[];
  classes: string[];
  num_classes: number;
}

// Cache for loaded data
let cachedModel: XGBoostModel | null = null;
let cachedMetadata: ModelMetadata | null = null;

/**
 * Fetch and parse XGBoost booster data
 * @returns Promise that resolves to parsed XGBoost model
 */
export async function loadXGBoostModel(): Promise<XGBoostModel> {
  if (cachedModel) {
    return cachedModel;
  }

  try {
    console.log('Loading actual XGBoost model from server...');
    
    // Determine the correct path based on environment
    let modelUrl = '/xgboost_model.json';
    
    // For GitHub Pages deployment with basePath
    if (typeof window !== 'undefined') {
      const pathPrefix = window.location.pathname.includes('/se-project') ? '/se-project' : '';
      modelUrl = `${pathPrefix}/xgboost_model.json`;
    }
    
    console.log(`Fetching model from: ${modelUrl}`);
    
    // Fetch the actual model file
    const modelResponse = await fetch(modelUrl);
    if (!modelResponse.ok) {
      throw new Error(`Failed to fetch model: ${modelResponse.statusText} (URL: ${modelUrl})`);
    }
    
    cachedModel = await modelResponse.json();
    console.log('✓ Real XGBoost model loaded successfully');
    console.log(`  - Total trees: ${cachedModel?.learner.gradient_booster.model.trees.length}`);
    console.log(`  - Features: ${cachedModel?.learner.feature_names.length}`);
    console.log(`  - Classes: ${cachedModel?.learner.objective.softmax_multiclass_param.num_class}`);
    
    return cachedModel!;
  } catch (error) {
    console.error('Failed to load XGBoost model:', error);
    throw new Error(`Model loading failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

/**
 * Load model metadata
 * @returns Promise that resolves to model metadata
 */
export async function loadModelMetadata(): Promise<ModelMetadata> {
  if (cachedMetadata) {
    return cachedMetadata;
  }

  try {
    // Determine the correct path based on environment
    let metadataUrl = '/model_metadata.json';
    
    // For GitHub Pages deployment with basePath
    if (typeof window !== 'undefined') {
      const pathPrefix = window.location.pathname.includes('/se-project') ? '/se-project' : '';
      metadataUrl = `${pathPrefix}/model_metadata.json`;
    }
    
    const metadataResponse = await fetch(metadataUrl);
    if (!metadataResponse.ok) {
      throw new Error(`Failed to fetch metadata: ${metadataResponse.statusText} (URL: ${metadataUrl})`);
    }
    
    cachedMetadata = await metadataResponse.json();
    console.log('✓ Model metadata loaded successfully');
    return cachedMetadata!;
  } catch (error) {
    console.error('Failed to load metadata:', error);
    throw new Error(`Metadata loading failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}


/**
 * Traverse a single XGBoost tree using the actual tree structure
 * @param tree - XGBoost tree to traverse
 * @param features - Input features
 * @returns Leaf value (weight)
 */
export function traverseTree(tree: XGBoostTree, features: number[]): number {
  // Real XGBoost tree traversal
  // Trees use array-based representation where each index is a node
  
  const leftChildren = tree.left_children || [];
  const rightChildren = tree.right_children || [];
  const splitIndices = tree.split_indices || [];
  const splitConditions = tree.split_conditions || [];
  const baseWeights = tree.base_weights || [];
  
  // Start at root node (index 0)
  let nodeIdx = 0;
  
  // Traverse until we reach a leaf node
  while (leftChildren[nodeIdx] !== -1 && rightChildren[nodeIdx] !== -1) {
    const featureIdx = splitIndices[nodeIdx];
    const threshold = splitConditions[nodeIdx];
    const featureValue = features[featureIdx];
    
    // Decide which child to go to
    if (featureValue < threshold) {
      nodeIdx = leftChildren[nodeIdx];
    } else {
      nodeIdx = rightChildren[nodeIdx];
    }
    
    // Safety check to prevent infinite loops
    if (nodeIdx < 0 || nodeIdx >= leftChildren.length) {
      console.warn('Invalid node index during tree traversal');
      break;
    }
  }
  
  // Return the leaf weight
  return baseWeights[nodeIdx] || 0;
}

/**
 * Get prediction from XGBoost model using actual tree ensemble
 * @param model - Parsed XGBoost model
 * @param features - Input features
 * @returns Class probabilities
 */
export function getXGBoostPrediction(model: XGBoostModel, features: number[]): number[] {
  const numClasses = parseInt(model.learner.objective.softmax_multiclass_param.num_class);
  const trees = model.learner.gradient_booster.model.trees;
  const treeInfo = model.learner.gradient_booster.model.tree_info;
  const baseScore = parseFloat(model.learner.learner_model_param.base_score);
  
  // Initialize margin for each class with base score
  const margins = new Array(numClasses).fill(baseScore);
  
  // For multi-class softprob, trees are grouped by class
  // tree_info tells us which class each tree belongs to
  for (let i = 0; i < trees.length; i++) {
    const tree = trees[i];
    const classIdx = treeInfo[i];
    
    // Traverse this tree and add its contribution to the class margin
    const treeOutput = traverseTree(tree, features);
    margins[classIdx] += treeOutput;
  }
  
  // Apply softmax to convert margins to probabilities
  const maxMargin = Math.max(...margins);
  const expMargins = margins.map(margin => Math.exp(margin - maxMargin));
  const sumExpMargins = expMargins.reduce((sum, val) => sum + val, 0);
  const probabilities = expMargins.map(val => val / sumExpMargins);
  
  return probabilities;
}