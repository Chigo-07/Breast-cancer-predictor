import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# 1. Configuration & Data Loading
# ==========================================
# Define the path to your dataset
DATA_PATH = 'data.csv'  # Ensure this file is in the same directory
MODEL_SAVE_PATH = 'breast_cancer_model.pkl'

print("Loading dataset...")

if not os.path.exists(DATA_PATH):
    print(f"Error: '{DATA_PATH}' not found. Please download the Breast Cancer Wisconsin dataset.")
    exit()

df = pd.read_csv(DATA_PATH)

# ==========================================
# 2. Feature Selection & Preprocessing
# ==========================================
# Features selected based on Project 5 requirements
selected_features = [
    'radius_mean', 
    'perimeter_mean', 
    'area_mean', 
    'compactness_mean', 
    'concavity_mean'
]
target_col = 'diagnosis'

# Validation: Check if columns exist
missing_cols = [col for col in selected_features + [target_col] if col not in df.columns]
if missing_cols:
    print(f"Error: The following columns are missing from the CSV: {missing_cols}")
    exit()

# Separate features (X) and target (y)
X = df[selected_features]
y = df[target_col]

# Encode Target: Malignant (M) -> 1, Benign (B) -> 0
# This is crucial for binary classification
y = y.map({'M': 1, 'B': 0})

# Split the data (80% Training, 20% Validation)
# This addresses the "Validation Step" feedback
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Data Split: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples.")

# ==========================================
# 3. Pipeline Construction (Preprocessing + Model)
# ==========================================
# We use a Pipeline to bundle the Scaler and the Model.
# This fixes the "Data Preprocessing" feedback ensuring consistency.
pipeline = Pipeline([
    ('scaler', StandardScaler()),              # 1. Standardize features
    ('classifier', LogisticRegression())       # 2. Apply Logistic Regression
])

# ==========================================
# 4. Training
# ==========================================
print("Training the model...")
pipeline.fit(X_train, y_train)

# ==========================================
# 5. Evaluation
# ==========================================
# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Calculate metrics
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\n" + "="*40)
print(f"Model Accuracy: {acc:.4f}")
print("="*40)
print("\nClassification Report:\n")
print(report)

# ==========================================
# 6. Save the Model
# ==========================================
# Save the ENTIRE pipeline (Scaler + Model)
joblib.dump(pipeline, MODEL_SAVE_PATH)

print(f"\nSuccess! Model saved to: {os.path.abspath(MODEL_SAVE_PATH)}")
print("You can now load this file in app.py to make predictions.")