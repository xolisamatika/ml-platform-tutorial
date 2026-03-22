"""
Train fraud detection model with MLflow experiment tracking.

This script demonstrates proper ML experiment tracking:
- Log all hyperparameters
- Log all metrics (train and test)
- Log the trained model as an artifact
- Register the model in the Model Registry

Compare this to train_naive.py to see the difference!
"""
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score
)
import pickle
from datetime import datetime
import os

# Configure MLflow to use our tracking server
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))

# Create or get the experiment
# All runs will be grouped under this experiment name
mlflow.set_experiment("fraud-detection")

def load_and_preprocess_data():
    """Load and preprocess the training and test data."""
    print("Loading data...")
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    
    # Encode categorical feature
    encoder = LabelEncoder()
    train_df["merchant_encoded"] = encoder.fit_transform(train_df["merchant_category"])
    test_df["merchant_encoded"] = encoder.transform(test_df["merchant_category"])
    
    # Prepare features
    feature_cols = ["amount", "hour", "day_of_week", "merchant_encoded"]
    X_train = train_df[feature_cols]
    y_train = train_df["is_fraud"]
    X_test = test_df[feature_cols]
    y_test = test_df["is_fraud"]
    
    return X_train, y_train, X_test, y_test, encoder

def train_and_log_model(
    n_estimators: int = 100,
    max_depth: int = 10,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1
):
    """
    Train a model and log everything to MLflow.
    
    Args:
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of each tree
        min_samples_split: Minimum samples required to split a node
        min_samples_leaf: Minimum samples required at a leaf node
    """
    X_train, y_train, X_test, y_test, encoder = load_and_preprocess_data()
    
    # Start an MLflow run - everything logged will be associated with this run
    with mlflow.start_run():
        # Add a descriptive run name
        run_name = f"rf_est{n_estimators}_depth{max_depth}_{datetime.now().strftime('%H%M%S')}"
        mlflow.set_tag("mlflow.runName", run_name)
        
        # Log all hyperparameters
        # These are the "knobs" we can tune
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)
        mlflow.log_param("model_type", "RandomForestClassifier")
        
        # Log data information
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("fraud_ratio", float(y_train.mean()))
        mlflow.log_param("n_features", X_train.shape[1])
        
        # Train the model
        print(f"\nTraining model: n_estimators={n_estimators}, max_depth={max_depth}")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Evaluate and log metrics for BOTH train and test sets
        # This helps detect overfitting
        for dataset_name, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X)[:, 1]
            
            # Calculate all metrics
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, zero_division=0)
            recall = recall_score(y, y_pred, zero_division=0)
            f1 = f1_score(y, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y, y_prob)
            
            # Log metrics with dataset prefix
            mlflow.log_metric(f"{dataset_name}_accuracy", accuracy)
            mlflow.log_metric(f"{dataset_name}_precision", precision)
            mlflow.log_metric(f"{dataset_name}_recall", recall)
            mlflow.log_metric(f"{dataset_name}_f1", f1)
            mlflow.log_metric(f"{dataset_name}_roc_auc", roc_auc)
            
            print(f"  {dataset_name.upper()} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
        
        # Log feature importance
        for feature, importance in zip(
            ["amount", "hour", "day_of_week", "merchant_encoded"],
            model.feature_importances_
        ):
            mlflow.log_metric(f"importance_{feature}", importance)
        
        # Log the model to MLflow AND register it in the Model Registry
        # This creates a new version of the model automatically
        print("\nRegistering model in MLflow Model Registry...")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="fraud-detection-model",
            input_example=X_train.iloc[:5]  # Example input for documentation
        )
        
        # Save and log the encoder as a separate artifact
        # We need this for inference
        with open("encoder.pkl", "wb") as f:
            pickle.dump(encoder, f)
        mlflow.log_artifact("encoder.pkl")
        
        # Get the run ID for reference
        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow Run ID: {run_id}")
        print(f"View this run: http://localhost:5000/#/experiments/1/runs/{run_id}")
        
        return model, encoder

def run_experiment_sweep():
    """
    Run multiple experiments with different hyperparameters.
    
    This demonstrates how MLflow helps compare different configurations.
    """
    print("="*60)
    print("RUNNING HYPERPARAMETER EXPERIMENT SWEEP")
    print("="*60)
    
    # Define different configurations to try
    experiments = [
        {"n_estimators": 50, "max_depth": 5},
        {"n_estimators": 100, "max_depth": 10},
        {"n_estimators": 100, "max_depth": 15},
        {"n_estimators": 200, "max_depth": 10},
        {"n_estimators": 200, "max_depth": 20},
    ]
    
    for i, params in enumerate(experiments, 1):
        print(f"\n--- Experiment {i}/{len(experiments)} ---")
        train_and_log_model(**params)
    
    print("\n" + "="*60)
    print("EXPERIMENT SWEEP COMPLETE!")
    print("="*60)
    print("\nView all experiments at: http://localhost:5000")
    print("Compare runs to find the best hyperparameters!")

if __name__ == "__main__":
    run_experiment_sweep()