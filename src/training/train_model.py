# Import necessary libraries
import os
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.tensorflow  # <-- ADDED for TensorFlow model logging
import logging
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import (
    StandardScaler,
)  # <-- ADDED for scaling data for the neural network

# Import ML frameworks
import xgboost as xgb
import tensorflow as tf  # <-- ADDED

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def setup_arg_parser():
    """Sets up the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Model Training Script for Fraud Detection"
    )

    # --- General arguments ---
    parser.add_argument(
        "--data_version",
        type=str,
        default="sample_1000000",
        help="Version of the data to use (e.g., 'sample_1000000' or 'full').",
    )
    parser.add_argument(
        "--run_name", type=str, default=None, help="Name for the MLflow run."
    )
    parser.add_argument(
        "--random_state", type=int, default=50, help="Random state for reproducibility."
    )

    # --- Model Selection argument --- (updated)
    parser.add_argument(
        "--model_type",
        type=str,
        default="xgboost",
        choices=["xgboost", "tensorflow_mlp"],  # <-- ADDED "tensorflow_mlp"
        help="Type of model to train.",
    )

    # --- XGBoost Specific Hyperparameters ---
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="Number of estimators for XGBoost.",
    )

    # --- TensorFlow Specific Hyperparameters --- (updated)
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer (used by both models).",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs for TensorFlow model."
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for TensorFlow model."
    )

    return parser


def load_data(data_version):
    """Loads the Spark-processed and DVC-managed data."""
    data_dir = f"data/processed/engineered_features_{data_version}"
    logging.info(f"Loading data from '{data_dir}'...")
    if not os.path.exists(data_dir):
        error_msg = (
            f"Processed data directory not found at '{data_dir}'.\n"
            f"Ensure you ran the Spark preprocessing script and (if using 'full') 'dvc pull'."
        )
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)

    df = pd.read_parquet(data_dir)
    logging.info(f"Data loaded successfully. Shape: {df.shape}")
    return df


# --- NEW FUNCTION FOR TENSORFLOW ---
def train_tensorflow_mlp(X_train, y_train, X_test, y_test, args):
    """Defines, trains, and evaluates a TensorFlow MLP model."""
    logging.info("Starting TensorFlow MLP model training...")

    # Step 1: Data Preprocessing for Neural Networks (Scaling)
    logging.info("Applying StandardScaler to features for the neural network.")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # CRITICAL: Log the scaler. It's part of the "model" now.
    mlflow.sklearn.log_model(scaler, "standard_scaler")
    logging.info("StandardScaler fitted and logged to MLflow as 'standard_scaler'.")

    # Step 2: Define the Keras Model Architecture
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
            tf.keras.layers.Dense(
                128, activation="relu", kernel_initializer="he_normal"
            ),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(
                64, activation="relu", kernel_initializer="he_normal"
            ),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(
                1, activation="sigmoid"
            ),  # Sigmoid for binary probability output
        ]
    )
    model.summary(print_fn=logging.info)

    # Step 3: Compile the Model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    # Step 4: Train the Model
    logging.info(f"Training TensorFlow model for {args.epochs} epochs...")
    model.fit(
        X_train_scaled,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_test_scaled, y_test),
        verbose=2,  # Verbose=2 shows one line per epoch
    )

    # Step 5: Evaluate and Log Final Metrics
    logging.info("Evaluating final TensorFlow model on test set...")
    pred_proba = model.predict(X_test_scaled).flatten()
    preds = (pred_proba > 0.5).astype(int)
    final_metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "f1_score": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, pred_proba),
    }
    mlflow.log_metrics(final_metrics)
    logging.info(f"Logged final TensorFlow metrics: {final_metrics}")

    # Step 6: Log the TensorFlow Model to MLflow
    mlflow.tensorflow.log_model(model, "tensorflow-mlp-model")
    logging.info("TensorFlow MLP model logged to MLflow as 'tensorflow-mlp-model'.")


# --- EXISTING FUNCTION FOR XGBOOST (with minor edits for clarity) ---
def train_xgboost(X_train, y_train, X_test, y_test, args):
    """Trains and evaluates an XGBoost model."""
    logging.info("Training XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        random_state=args.random_state,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    logging.info("Model training complete.")

    logging.info("Evaluating XGBoost model...")
    preds = model.predict(X_test)
    pred_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "f1_score": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, pred_proba),
    }
    mlflow.log_metrics(metrics)
    logging.info(f"Logged XGBoost metrics: {metrics}")

    mlflow.xgboost.log_model(model, "xgboost-model")
    logging.info("XGBoost model logged to MLflow as 'xgboost-model'.")


def main(args):
    """Main function to orchestrate the model training and logging pipeline."""
    with mlflow.start_run(run_name=args.run_name):
        logging.info(
            f"Starting MLflow run: '{args.run_name}' for model type: '{args.model_type}'"
        )

        # Log configuration
        mlflow.log_params(vars(args))
        mlflow.set_tag("data_version", args.data_version)
        mlflow.set_tag("model_type", args.model_type)
        logging.info(f"Logged parameters: {vars(args)}")

        # Data Loading & Splitting
        df = load_data(args.data_version)
        target_column = "isFraud"
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found.")

        X = df.drop(target_column, axis=1)
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=args.random_state, stratify=y
        )
        logging.info(
            f"Data split. Train shape: {X_train.shape}, Test shape: {X_test.shape}"
        )

        # --- Model Training based on choice --- (updated)
        if args.model_type == "xgboost":
            train_xgboost(X_train, y_train, X_test, y_test, args)
        elif args.model_type == "tensorflow_mlp":
            train_tensorflow_mlp(X_train, y_train, X_test, y_test, args)

        logging.info(f"MLflow run '{args.run_name}' finished successfully.")


if __name__ == "__main__":
    arg_parser = setup_arg_parser()
    cli_args = arg_parser.parse_args()
    main(cli_args)
