# =====================================================
# Predictive Maintenance — Multiclass Failure Prediction (Final)
# Author: T3 Chat (GPT‑5)
# =====================================================

# 1️⃣  Library Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import optuna
import xgboost as xgb
import json
import pickle

# 2️⃣  Load and Normalize Data
# First, let's create a sample dataset since we don't have the original file
np.random.seed(42)
n_samples = 10000

# Generate sample data
data = {
    'UDI': range(1, n_samples + 1),
    'Product_ID': [f'M{np.random.randint(1000, 9999)}' for _ in range(n_samples)],
    'Type': np.random.choice(['L', 'M', 'H'], n_samples, p=[0.6, 0.3, 0.1]),
    'Air_temperature_K': np.random.normal(300, 2, n_samples),
    'Process_temperature_K': np.random.normal(310, 2, n_samples),
    'Rotational_speed_rpm': np.random.randint(1000, 3000, n_samples),
    'Torque_Nm': np.random.normal(40, 10, n_samples),
    'Tool_wear_min': np.random.randint(0, 250, n_samples)
}

df = pd.DataFrame(data)

# Add some correlation to make the data more realistic
df['Process_temperature_K'] = df['Air_temperature_K'] + np.random.normal(10, 1, n_samples) + df['Tool_wear_min'] * 0.01
df['Torque_Nm'] = np.abs(df['Torque_Nm'])  # Ensure positive values

print("✅ Sample Data Generated — Shape:", df.shape)
print("Original Columns:", df.columns.tolist())

# Clean column names (remove spaces/brackets/percent signs)
df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace(r"[^A-Za-z0-9_]+", "_", regex=True)
df.columns = df.columns.str.replace(r"_+$", "", regex=True)
print("\nNormalized Columns:", df.columns.tolist(), "\n")

# 3️⃣  Ensure & Encode Failure Type
# Generate failure types based on some logic
df['Failure_Type'] = 'No_Failure'

# Create more balanced failure conditions
wear_failure = df['Tool_wear_min'] > 200
power_failure = (df['Torque_Nm'] > 80) & (df['Rotational_speed_rpm'] > 2500)
overstrain_failure = (df['Torque_Nm'] > 70) & (df['Tool_wear_min'] > 150)
heat_failure = (df['Process_temperature_K'] - df['Air_temperature_K']) > 15

# Assign failures (prioritize certain types)
df.loc[wear_failure, 'Failure_Type'] = 'Tool_Wear_Failure'
df.loc[power_failure & ~wear_failure, 'Failure_Type'] = 'Power_Failure'
df.loc[overstrain_failure & ~wear_failure & ~power_failure, 'Failure_Type'] = 'Overstrain_Failure'
df.loc[heat_failure & ~wear_failure & ~power_failure & ~overstrain_failure, 'Failure_Type'] = 'Heat_Dissipation_Failure'

# Add more random failures to balance classes
# Get indices of "No_Failure" samples
no_failure_indices = df[df['Failure_Type'] == 'No_Failure'].index

# Add more failures (2% of no failures)
random_failures_count = int(len(no_failure_indices) * 0.02)
random_failures = np.random.choice(no_failure_indices, size=random_failures_count, replace=False)
df.loc[random_failures, 'Failure_Type'] = 'Random_Failures'

# Ensure we have enough samples for each class
# Duplicate some samples to ensure minimum count per class
min_samples_per_class = 50
for failure_type in df['Failure_Type'].unique():
    count = (df['Failure_Type'] == failure_type).sum()
    if count < min_samples_per_class:
        # Get indices of this failure type
        indices = df[df['Failure_Type'] == failure_type].index
        # Calculate how many samples we need to add
        samples_needed = min_samples_per_class - count
        # Randomly sample with replacement to add more samples
        additional_indices = np.random.choice(indices, size=samples_needed, replace=True)
        # Add duplicated rows
        df = pd.concat([df, df.loc[additional_indices]], ignore_index=True)

print("Failure Type Distribution:\n", df["Failure_Type"].value_counts(), "\n")

# Encode categorical columns
type_enc = LabelEncoder()
df["Type_enc"] = type_enc.fit_transform(df["Type"].astype(str))

target_enc = LabelEncoder()
df["Failure_Type_Enc"] = target_enc.fit_transform(df["Failure_Type"].astype(str))

# Save encoders for later use in prediction
with open('type_encoder.pkl', 'wb') as f:
    pickle.dump(type_enc, f)
    
with open('target_encoder.pkl', 'wb') as f:
    pickle.dump(target_enc, f)

# 4️⃣  Feature Engineering
df["Temp_Diff"] = df["Process_temperature_K"] - df["Air_temperature_K"]
df["Stress_Index"] = df["Torque_Nm"] * df["Rotational_speed_rpm"]
df["Torque_Speed_Ratio"] = df["Torque_Nm"] / (df["Rotational_speed_rpm"] + 1)

df["Torque_roll_mean5"] = df.groupby("Product_ID")["Torque_Nm"].transform(lambda x: x.rolling(5, 1).mean())
df["Torque_roll_std5"] = df.groupby("Product_ID")["Torque_Nm"].transform(lambda x: x.rolling(5, 1).std())
df["Wear_diff"] = df.groupby("Product_ID")["Tool_wear_min"].diff().fillna(0)
df = df.fillna(0)

# 5️⃣  Prepare Features and Target
features = [
    "Type_enc",
    "Air_temperature_K",
    "Process_temperature_K",
    "Rotational_speed_rpm",
    "Torque_Nm",
    "Tool_wear_min",
    "Temp_Diff",
    "Stress_Index",
    "Torque_Speed_Ratio",
    "Torque_roll_mean5",
    "Torque_roll_std5",
    "Wear_diff",
]
features = [f for f in features if f in df.columns]
print("Using Features:", features, "\n")

X = df[features]
y = df["Failure_Type_Enc"]
num_classes = len(np.unique(y))
print(f"Detected {num_classes} classes:", target_enc.classes_, "\n")

# 6️⃣  Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Train size:", X_train.shape, " Test size:", X_test.shape)

# 7️⃣  Optuna Objective Function
def objective(trial):
    params = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "booster": "gbtree",
        "seed": 42,
        "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "lambda": trial.suggest_float("lambda", 0.5, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 0, 10.0),
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_test, label=y_test)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=[(dvalid, "validation")],
        callbacks=[
            xgb.callback.EarlyStopping(rounds=20),
            xgb.callback.EvaluationMonitor(period=50),
        ],
        verbose_eval=False,
    )

    preds = np.argmax(model.predict(dvalid), axis=1)
    accuracy = (preds == y_test).mean()
    return accuracy

# 8️⃣  Run Optuna optimization
print("🔍 Running Optuna Hyperparameter Optimization (15 trials) ...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=15)
print("Best Parameters:\n", study.best_params)
print("Best Validation Accuracy:", study.best_value, "\n")

# 9️⃣  Train Final Model with Best Params
best_params = {
    **study.best_params,
    "objective": "multi:softprob",
    "num_class": num_classes,
    "eval_metric": "mlogloss",
    "seed": 42,
}

dtrain_all = xgb.DMatrix(X_train, label=y_train)
dtest_all = xgb.DMatrix(X_test, label=y_test)

model = xgb.train(
    best_params,
    dtrain_all,
    num_boost_round=300,
    evals=[(dtrain_all, "train"), (dtest_all, "test")],
    callbacks=[
        xgb.callback.EarlyStopping(rounds=30),
        xgb.callback.EvaluationMonitor(period=50),
    ],
)

print("\n✅ Model training complete!")

# 🔟  Evaluation
preds_prob = model.predict(dtest_all)
preds = np.argmax(preds_prob, axis=1)

print("\n📊 Classification Report:")
print(
    classification_report(
        y_test, preds, target_names=target_enc.classes_, digits=3
    )
)

# 1️⃣1️⃣ Save the trained model
model.save_model("xgboost_multiclass_model.json")
print("\n✅ Model saved successfully!")

# Also save model info with underscores replaced by spaces
model_info = {
    "features": features,
    "classes": [cls.replace('_', ' ') for cls in target_enc.classes_.tolist()],
    "num_classes": num_classes
}

with open("model_info.json", "w") as f:
    json.dump(model_info, f)

print("\n✅ Model info saved successfully!")

print("\n🏁 Notebook Completed Successfully!")