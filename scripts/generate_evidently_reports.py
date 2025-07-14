# scripts/generate_evidently_reports.py (Final, Corrected, and Professional Version)

import os
import pandas as pd
import mlflow
import logging
from sklearn.model_selection import train_test_split

# --- CORRECT IMPORTS ---
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently.pipeline.column_mapping import ColumnMapping

# --- Configuration ---
# This section contains all the user-configurable settings for the script.

# 1. MLflow Configuration:
MLFLOW_RUN_ID = "00b7510a863e47e88ad4a63a9297e6b2" # Your champion XGBoost run ID
MODEL_ARTIFACT_PATH = "xgboost-model"                # The artifact path from the MLflow UI

# 2. Path Configuration:
PROCESSED_DATA_PATH = "data/processed/engineered_features_full" # Path to the input data
REPORT_OUTPUT_DIR = "docs/evidently_reports"                  # Directory to save the reports

# 3. Script Configuration:
TARGET_COLUMN = 'isFraud'
RANDOM_STATE = 50 # Must match the state used in your training script

# --- Script Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_model_and_get_predictions(run_id, model_path, features_df):
    """Loads a model from an MLflow run and generates predictions."""
    try:
        model_uri = f"runs:/{run_id}/{model_path}"
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        logging.info(f"Model loaded successfully from {model_uri}")

        # This will be used for the performance report
        predictions = loaded_model.predict(features_df)
        return predictions
    except Exception as e:
        logging.error(f"Error in load_model_and_get_predictions: {e}")
        raise

def main():
    """Main function to generate and save Evidently AI reports."""
    logging.info("Starting Evidently AI report generation...")
    os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)

    logging.info(f"Loading full processed data from '{PROCESSED_DATA_PATH}'...")
    if not os.path.exists(PROCESSED_DATA_PATH):
        logging.error(f"Data not found at {PROCESSED_DATA_PATH}. Please run 'dvc pull' first.")
        return
    
    df = pd.read_parquet(PROCESSED_DATA_PATH)
    
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    logging.info("Data loaded and split successfully.")
    
    # --- Prepare Datasets for Evidently ---
    # The 'reference' data is the data the model was trained on.
    reference_data = X_train.copy()
    reference_data[TARGET_COLUMN] = y_train

    # The 'current' data is new data we want to evaluate. We'll use our test set.
    current_data = X_test.copy()
    current_data[TARGET_COLUMN] = y_test
    
    # --- Generate Data Drift Report ---
    # This report only needs the features and the true target. It does not need predictions.
    logging.info("Generating Data Drift report...")
    data_drift_report = Report(metrics=[DataDriftPreset()])
    # The .run() method calculates the metrics
    data_drift_report.run(reference_data=reference_data, current_data=current_data)
    drift_report_path = os.path.join(REPORT_OUTPUT_DIR, "data_drift_report.html")
    # The report object itself has the .save_html() method in modern versions
    data_drift_report.save_html(drift_report_path)
    logging.info(f"Data Drift report saved to {drift_report_path}")

    # --- Generate Classification Performance Report ---
    logging.info(f"Generating Classification Performance report using model from run ID: {MLFLOW_RUN_ID}")
    
    # Add predictions to our datasets for the performance report
    current_predictions = load_model_and_get_predictions(MLFLOW_RUN_ID, MODEL_ARTIFACT_PATH, X_test)
    current_data['prediction'] = current_predictions

    reference_predictions = load_model_and_get_predictions(MLFLOW_RUN_ID, MODEL_ARTIFACT_PATH, X_train)
    reference_data['prediction'] = reference_predictions

    # A ColumnMapping is needed to tell Evidently the roles of the columns
    column_mapping = ColumnMapping(
        target=TARGET_COLUMN,
        prediction='prediction',
    )
    
    classification_report = Report(metrics=[ClassificationPreset()])
    classification_report.run(
        reference_data=reference_data, 
        current_data=current_data,
        column_mapping=column_mapping # Pass the mapping here
    )
    performance_report_path = os.path.join(REPORT_OUTPUT_DIR, "model_performance_report.html")
    classification_report.save_html(performance_report_path)
    logging.info(f"Model Performance report saved to {performance_report_path}")
            
    logging.info("Evidently AI report generation finished successfully.")

if __name__ == "__main__":
    main()