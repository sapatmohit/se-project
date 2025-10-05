# Predictive Maintenance Dashboard

A full-stack Next.js 15 web interface for an AI-powered predictive maintenance system. This application uses machine learning to predict equipment failures based on sensor data.

## Features

- Real-time failure prediction using XGBoost machine learning model
- Responsive dashboard with Tailwind CSS styling
- Sensor data input form (Air temperature, Process temperature, Rotational speed, Torque, Tool wear, Machine Type)
- Color-coded prediction results with confidence levels
- Prediction history tracking
- REST API for model inference

## Prerequisites

- Node.js 18+
- Python 3.8+
- npm or yarn

## Getting Started

### 1. Install Node.js dependencies

```bash
npm install
```

### 2. Install Python dependencies

```bash
pip install -r AI/requirements.txt
```

### 3. Train the model

```bash
python AI/model_train.py
```

This will generate the model file and encoders needed for predictions.

### 4. Run the development server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the application.

## Project Structure

```
.
├── AI/                    # Machine learning models and scripts
│   ├── model_train.py     # Model training script
│   ├── predict.py         # Prediction script for API
│   ├── requirements.txt   # Python dependencies
│   └── xgboost_multiclass_model.json  # Trained model (generated after training)
├── src/
│   ├── app/               # Next.js App Router
│   │   ├── api/predict/   # Prediction API endpoint
│   │   ├── page.tsx       # Main dashboard page
│   │   ├── layout.tsx     # Root layout
│   │   └── globals.css    # Global styles
├── package.json           # Node.js dependencies
└── README.md              # This file
```

## How It Works

1. **Model Training**: The `model_train.py` script generates a synthetic dataset and trains an XGBoost classifier to predict equipment failures.

2. **Prediction API**: The `/api/predict` endpoint receives sensor data, processes it through the trained model, and returns the predicted failure type with confidence.

3. **Frontend Dashboard**: The React-based dashboard allows users to input sensor readings and view predictions in real-time.

## Customization

### Modifying the Model

To retrain the model with different parameters, modify `AI/model_train.py` and run:

```bash
python AI/model_train.py
```

### Updating the Dashboard

The main dashboard is located at `src/app/page.tsx`. You can modify the UI components and styling there.

## Learn More

To learn more about the technologies used:

- [Next.js Documentation](https://nextjs.org/docs)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Tailwind CSS Documentation](https://tailwindcss.com/docs)