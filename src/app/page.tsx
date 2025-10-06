'use client';

import { useEffect, useRef, useState } from 'react';
import { FaCheckCircle, FaExclamationTriangle, FaHistory, FaInfoCircle, FaPlay, FaTachometerAlt, FaThermometerHalf, FaTools, FaTrash, FaUpload } from 'react-icons/fa';

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

// Types for our prediction history
interface PredictionHistoryItem extends PredictionResponse {
  id: string;
  inputs: PredictionInput;
}

export default function PredictiveMaintenanceDashboard() {
  // Form state
  const [formData, setFormData] = useState<PredictionInput>({
    airTemperature: 298.15,
    processTemperature: 308.15,
    rotationalSpeed: 1500,
    torque: 40,
    toolWear: 0,
    type: 'L'
  });

  // File upload state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Prediction state
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // History state
  const [history, setHistory] = useState<PredictionHistoryItem[]>([]);

  // Load history from localStorage on component mount
  useEffect(() => {
    const savedHistory = localStorage.getItem('predictionHistory');
    if (savedHistory) {
      try {
        setHistory(JSON.parse(savedHistory));
      } catch (e) {
        console.error('Failed to parse history', e);
      }
    }
  }, []);

  // Save history to localStorage
  useEffect(() => {
    localStorage.setItem('predictionHistory', JSON.stringify(history));
  }, [history]);

  // Handle form input changes
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: name === 'type' ? value : parseFloat(value) || 0
    });
  };

  // Handle file selection
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setSelectedFile(e.target.files[0]);
    }
  };

  // Handle drag and drop events
  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setSelectedFile(e.dataTransfer.files[0]);
    }
  };

  // Handle form submission (manual input)
  const handleSubmitManual = async (e: React.FormEvent) => {
    e.preventDefault();
    await makePrediction('manual');
  };

  // Handle file submission
  const handleSubmitFile = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) {
      setError('Please select a JSON file');
      return;
    }
    await makePrediction('file');
  };

  // Make prediction based on input type
  const makePrediction = async (inputType: 'manual' | 'file') => {
    setIsLoading(true);
    setError(null);
    setPrediction(null);

    try {
      let response: Response;
      
      if (inputType === 'file' && selectedFile) {
        // File upload approach
        const formDataObj = new FormData();
        formDataObj.append('file', selectedFile);
        
        response = await fetch('/api/predict', {
          method: 'POST',
          body: formDataObj,
        });
      } else {
        // Manual input approach
        response = await fetch('/api/predict', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(formData),
        });
      }

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `Failed to get prediction: ${response.status}`);
      }

      const result: PredictionResponse = await response.json();
      setPrediction(result);

      // Add to history
      const historyItem: PredictionHistoryItem = {
        id: Date.now().toString(),
        ...result,
        inputs: inputType === 'file' ? formData : { ...formData }
      };

      setHistory(prev => [historyItem, ...prev].slice(0, 10)); // Keep only last 10 predictions
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
      console.error('Prediction error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  // Clear prediction history
  const clearHistory = () => {
    if (confirm('Are you sure you want to clear all prediction history? This action cannot be undone.')) {
      setHistory([]);
      localStorage.removeItem('predictionHistory');
    }
  };

  // Trigger file input click
  const triggerFileInput = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  // Clear selected file
  const clearFile = () => {
    setSelectedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Get confidence color based on value
  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'confidence-high';
    if (confidence >= 0.6) return 'confidence-medium';
    if (confidence >= 0.4) return 'confidence-medium';
    return 'confidence-low';
  };

  // Get failure type color
  const getFailureTypeColor = (failureType: string) => {
    switch (failureType) {
      case 'No Failure':
        return 'failure-type-no-failure';
      case 'Tool Wear Failure':
        return 'failure-type-tool-wear';
      case 'Power Failure':
        return 'failure-type-power';
      case 'Overstrain Failure':
        return 'failure-type-overstrain';
      case 'Heat Dissipation Failure':
        return 'failure-type-heat';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  // Get health indicator based on failure type
  const getHealthIndicator = (failureType: string) => {
    if (failureType === 'No Failure') {
      return <span className="health-indicator good"></span>;
    } else if (failureType.includes('Failure')) {
      return <span className="health-indicator critical"></span>;
    } else {
      return <span className="health-indicator warning"></span>;
    }
  };

  // Get icon for sensor type
  const getSensorIcon = (sensorType: string) => {
    switch (sensorType) {
      case 'temperature':
        return <FaThermometerHalf className="text-blue-500" />;
      case 'speed':
        return <FaTachometerAlt className="text-green-500" />;
      case 'torque':
        return <FaTools className="text-purple-500" />;
      default:
        return <FaExclamationTriangle className="text-yellow-500" />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 py-8 transition-colors duration-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-10">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full industrial-gradient mb-4">
            <FaTools className="text-white text-2xl" />
          </div>
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white sm:text-5xl">
            Predictive Maintenance
          </h1>
          <p className="mt-3 text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
            AI-powered system for monitoring machine health and predicting potential failures
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Input Form */}
          <div className="lg:col-span-1">
            <div className="bg-white dark:bg-gray-800 shadow-card rounded-card p-6 transition-colors duration-200">
              <div className="flex items-center mb-6">
                <div className="inline-flex items-center justify-center w-10 h-10 rounded-lg industrial-accent mr-3">
                  <FaTools className="text-white" />
                </div>
                <h2 className="text-2xl font-bold text-gray-800 dark:text-white">Machine Sensors</h2>
              </div>
              
              {/* Manual Input Form */}
              <form onSubmit={handleSubmitManual} className="space-y-5 mb-8">
                <h3 className="text-lg font-semibold text-gray-700 dark:text-gray-300">Manual Input</h3>
                
                {/* Air Temperature */}
                <div className="relative">
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1 flex items-center">
                    {getSensorIcon('temperature')}
                    <span className="ml-2">Air Temperature [K]</span>
                  </label>
                  <input
                    type="number"
                    name="airTemperature"
                    value={formData.airTemperature}
                    onChange={handleInputChange}
                    step="0.01"
                    className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent input-field bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    required
                  />
                </div>

                {/* Process Temperature */}
                <div className="relative">
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1 flex items-center">
                    {getSensorIcon('temperature')}
                    <span className="ml-2">Process Temperature [K]</span>
                  </label>
                  <input
                    type="number"
                    name="processTemperature"
                    value={formData.processTemperature}
                    onChange={handleInputChange}
                    step="0.01"
                    className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent input-field bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    required
                  />
                </div>

                {/* Rotational Speed */}
                <div className="relative">
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1 flex items-center">
                    {getSensorIcon('speed')}
                    <span className="ml-2">Rotational Speed [rpm]</span>
                  </label>
                  <input
                    type="number"
                    name="rotationalSpeed"
                    value={formData.rotationalSpeed}
                    onChange={handleInputChange}
                    className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent input-field bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    required
                  />
                </div>

                {/* Torque */}
                <div className="relative">
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1 flex items-center">
                    {getSensorIcon('torque')}
                    <span className="ml-2">Torque [Nm]</span>
                  </label>
                  <input
                    type="number"
                    name="torque"
                    value={formData.torque}
                    onChange={handleInputChange}
                    step="0.1"
                    className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent input-field bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    required
                  />
                </div>

                {/* Tool Wear */}
                <div className="relative">
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1 flex items-center">
                    <FaTools className="text-gray-500" />
                    <span className="ml-2">Tool Wear [min]</span>
                  </label>
                  <input
                    type="number"
                    name="toolWear"
                    value={formData.toolWear}
                    onChange={handleInputChange}
                    className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent input-field bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    required
                  />
                </div>

                {/* Machine Type */}
                <div className="relative">
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Machine Type
                  </label>
                  <select
                    name="type"
                    value={formData.type}
                    onChange={handleInputChange}
                    className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent input-field appearance-none bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  >
                    <option value="L">L (Low Complexity)</option>
                    <option value="M">M (Medium Complexity)</option>
                    <option value="H">H (High Complexity)</option>
                  </select>
                </div>

                <button
                  type="submit"
                  disabled={isLoading}
                  className="w-full flex justify-center py-3 px-4 border border-transparent rounded-lg shadow-sm text-sm font-medium text-white btn-primary disabled:opacity-50"
                >
                  {isLoading ? (
                    <span className="flex items-center">
                      <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Analyzing...
                    </span>
                  ) : (
                    <span className="flex items-center">
                      <FaPlay className="mr-2" />
                      Predict Failure (Manual)
                    </span>
                  )}
                </button>
              </form>

              {/* File Upload Form */}
              <div className="border-t border-gray-200 dark:border-gray-700 pt-6">
                <h3 className="text-lg font-semibold text-gray-700 dark:text-gray-300 mb-4">JSON File Input</h3>
                
                <div 
                  className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors ${
                    isDragging 
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' 
                      : 'border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500'
                  }`}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  onClick={triggerFileInput}
                >
                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileChange}
                    accept=".json"
                    className="hidden"
                  />
                  
                  <FaUpload className="mx-auto h-12 w-12 text-gray-400 dark:text-gray-500" />
                  <div className="mt-4">
                    <p className="text-lg font-medium text-gray-900 dark:text-white">
                      {selectedFile ? selectedFile.name : 'Drag and drop a JSON file'}
                    </p>
                    <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                      {selectedFile ? 'Click to change file' : 'or click to browse'}
                    </p>
                    <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                      JSON file should contain sensor data
                    </p>
                  </div>
                </div>
                
                {selectedFile && (
                  <div className="mt-4 flex justify-between">
                    <button
                      type="button"
                      onClick={clearFile}
                      className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 rounded-md hover:bg-gray-200 dark:hover:bg-gray-600"
                    >
                      Clear
                    </button>
                    <button
                      type="button"
                      onClick={handleSubmitFile}
                      disabled={isLoading}
                      className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 disabled:opacity-50 flex items-center"
                    >
                      {isLoading ? (
                        <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                      ) : (
                        <FaPlay className="mr-2" />
                      )}
                      Predict Failure (File)
                    </button>
                  </div>
                )}
              </div>

              {error && (
                <div className="mt-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
                  <div className="flex">
                    <div className="flex-shrink-0">
                      <FaExclamationTriangle className="h-5 w-5 text-red-400" />
                    </div>
                    <div className="ml-3">
                      <h3 className="text-sm font-medium text-red-800 dark:text-red-200">Error</h3>
                      <div className="mt-2 text-sm text-red-700 dark:text-red-300">
                        <p>{error}</p>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Results and History */}
          <div className="lg:col-span-2 space-y-8">
            {/* Prediction Result */}
            {prediction && (
              <div className="bg-white dark:bg-gray-800 shadow-card rounded-card p-6 transition-colors duration-200">
                <div className="flex items-center mb-6">
                  <div className="inline-flex items-center justify-center w-10 h-10 rounded-lg industrial-gradient mr-3">
                    <FaCheckCircle className="text-white" />
                  </div>
                  <h2 className="text-2xl font-bold text-gray-800 dark:text-white">Prediction Result</h2>
                </div>
                
                <div className="flex flex-col items-center justify-center py-8">
                  <div className="text-2xl font-bold mb-2 text-gray-700 dark:text-gray-300">Predicted Failure Type</div>
                  <div className={`text-4xl font-bold mb-6 px-8 py-4 rounded-full ${getFailureTypeColor(prediction.failureType)} shadow-lg`}>
                    {prediction.failureType}
                  </div>
                  
                  <div className="w-full max-w-md">
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-lg font-medium text-gray-700 dark:text-gray-300">Confidence Level</span>
                      <span className={`text-2xl font-bold ${getConfidenceColor(prediction.confidence)}`}>
                        {(prediction.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-4 mb-4">
                      <div 
                        className={`h-4 rounded-full ${prediction.confidence >= 0.8 ? 'bg-green-500' : prediction.confidence >= 0.6 ? 'bg-yellow-500' : prediction.confidence >= 0.4 ? 'bg-orange-500' : 'bg-red-500'}`}
                        style={{ width: `${prediction.confidence * 100}%` }}
                      ></div>
                    </div>
                    
                    <div className="flex justify-between text-sm text-gray-500 dark:text-gray-400">
                      <span>Low</span>
                      <span>High</span>
                    </div>
                  </div>
                  
                  <div className="mt-6 flex items-center text-gray-600 dark:text-gray-400">
                    {getHealthIndicator(prediction.failureType)}
                    <span className="ml-2">
                      Predicted at: {new Date(prediction.timestamp).toLocaleString()}
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* Prediction History */}
            <div className="bg-white dark:bg-gray-800 shadow-card rounded-card p-6 transition-colors duration-200">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center">
                  <div className="inline-flex items-center justify-center w-10 h-10 rounded-lg industrial-accent mr-3">
                    <FaHistory className="text-white" />
                  </div>
                  <h2 className="text-2xl font-bold text-gray-800 dark:text-white">Recent Predictions</h2>
                </div>
                {history.length > 0 && (
                  <button
                    onClick={clearHistory}
                    className="flex items-center px-3 py-2 text-sm font-medium text-red-700 dark:text-red-300 bg-red-50 dark:bg-red-900/20 rounded-md hover:bg-red-100 dark:hover:bg-red-900/30 transition-colors"
                  >
                    <FaTrash className="mr-1" />
                    Clear History
                  </button>
                )}
              </div>
              
              {history.length === 0 ? (
                <div className="text-center py-12">
                  <FaHistory className="mx-auto h-12 w-12 text-gray-400 dark:text-gray-500" />
                  <h3 className="mt-2 text-lg font-medium text-gray-900 dark:text-white">No predictions yet</h3>
                  <p className="mt-1 text-gray-500 dark:text-gray-400">
                    Submit sensor data to see your prediction history.
                  </p>
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                    <thead className="bg-gray-50 dark:bg-gray-700">
                      <tr>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                          Status
                        </th>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                          Failure Type
                        </th>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                          Confidence
                        </th>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                          Time
                        </th>
                      </tr>
                    </thead>
                    <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                      {history.map((item) => (
                        <tr key={item.id} className="hover:bg-gray-50 dark:hover:bg-gray-700">
                          <td className="px-6 py-4 whitespace-nowrap">
                            {getHealthIndicator(item.failureType)}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <span className={`px-3 py-1 inline-flex text-sm leading-5 font-semibold rounded-full ${getFailureTypeColor(item.failureType)}`}>
                              {item.failureType}
                            </span>
                          </td>
                          <td className={`px-6 py-4 whitespace-nowrap text-sm font-medium ${getConfidenceColor(item.confidence)}`}>
                            {(item.confidence * 100).toFixed(1)}%
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                            {new Date(item.timestamp).toLocaleTimeString()}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}