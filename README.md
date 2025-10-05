# Predictive Maintenance Dashboard

A full-stack Next.js 15 web interface for an AI-powered predictive maintenance system. This application uses machine learning to predict equipment failures based on sensor data.

## Features

- Real-time failure prediction using XGBoost machine learning model
- Responsive dashboard with Tailwind CSS styling
- Two input methods:
  - Manual sensor data input (Air temperature, Process temperature, Rotational speed, Torque, Tool wear, Machine Type)
  - JSON file upload for batch predictions
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
├── public/                # Static assets
│   └── sample-data.json   # Sample JSON file for testing
├── package.json           # Node.js dependencies
└── README.md              # This file
```

## How It Works

1. **Model Training**: The `model_train.py` script generates a synthetic dataset and trains an XGBoost classifier to predict equipment failures.

2. **Prediction API**: The `/api/predict` endpoint receives sensor data, processes it through the trained model, and returns the predicted failure type with confidence.

3. **Frontend Dashboard**: The React-based dashboard allows users to input sensor readings and view predictions in real-time.

## JSON File Input

You can upload a JSON file containing sensor data instead of manually entering values. The JSON file should have the following structure:

```json
{
  "airTemperature": 298.15,
  "processTemperature": 308.15,
  "rotationalSpeed": 1500,
  "torque": 40,
  "toolWear": 0,
  "type": "L"
}
```

A sample file is available at `/sample-data.json` in the public directory.

## Deployment

### GitHub Pages Deployment

This application can be deployed to GitHub Pages, but note that GitHub Pages only serves static files, so the prediction API will not be available in the deployed version.

To deploy to GitHub Pages:

1. Build and export the application:
   ```bash
   npm run deploy
   ```

2. Deploy the `out` directory to your GitHub Pages:
   - If using GitHub Actions, set up a workflow to automatically deploy the `out` directory
   - Or manually push the contents of the `out` directory to your `gh-pages` branch

The application will be accessible at: https://sapatmohit.github.io/se-project/

### Local Development with Full Functionality

To use the full functionality including the prediction API, run the application locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/sapatmohit/se-project.git
   cd se-project
   ```

2. Install dependencies:
   ```bash
   npm install
   pip install -r AI/requirements.txt
   ```

3. Train the model:
   ```bash
   python AI/model_train.py
   ```

4. Run the development server:
   ```bash
   npm run dev
   ```

5. Open [http://localhost:3000](http://localhost:3000) in your browser

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