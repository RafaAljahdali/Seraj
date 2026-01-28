import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Try importing XGBoost
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not found. Install with: pip install xgboost")

# Settings
DATA_PATH = "hospital_deterioration_ml_ready.csv"
TARGET_COL = "deterioration_next_12h"
MODEL_PATH = "model.joblib"
METRICS_PATH = "metrics.json"


def load_data():
    """Load the dataset"""
    print("Loading data...")
    try:
        df = pd.read_csv(DATA_PATH, sep=';')
        if len(df.columns) <= 1:
            df = pd.read_csv(DATA_PATH, sep=',')
    except:
        df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} records")
    print(f"Columns: {list(df.columns)}")
    return df


def prepare_data(df):
    """Prepare data for training"""
    print("Preparing data...")
    
    # Remove unnecessary columns
    exclude_cols = ['patient_id', 'encounter_id', 'unit', 'room', 'bed', 
                    'admission_date', 'record_timestamp']
    
    for col in exclude_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Separate features from target
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    
    # Identify column types
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"Number of features: {len(numeric_cols)}")
    return X, y, numeric_cols


def create_pipeline(numeric_cols):
    """Create training pipeline"""
    
    # Numeric data processing
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols)
        ],
        remainder='drop'
    )
    
    # Model
    if HAS_XGB:
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
    else:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        )
    
    # Full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    return pipeline


def train():
    """Train the model"""
    print("\n" + "="*50)
    print("PATIENT DETERIORATION PREDICTION - MODEL TRAINING")
    print("="*50 + "\n")
    
    # Load data
    df = load_data()
    
    # Prepare data
    X, y, numeric_cols = prepare_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {len(X_train)} records")
    print(f"Test set: {len(X_test)} records")
    
    # Create and train pipeline
    pipeline = create_pipeline(numeric_cols)
    print("\nTraining in progress...")
    pipeline.fit(X_train, y_train)
    print("Training complete!")
    
    # Evaluate model
    print("\n" + "-"*50)
    print("EVALUATION RESULTS:")
    print("-"*50)
    
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    
    # Save model
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved: {MODEL_PATH}")
    
    # Save metrics
    metrics = {
        "roc_auc": float(roc_auc),
        "f1_score": float(f1),
        "recall": float(recall),
        "precision": float(precision),
        "numeric_features": numeric_cols
    }
    
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {METRICS_PATH}")
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*50 + "\n")


if __name__ == "__main__":
    train()
