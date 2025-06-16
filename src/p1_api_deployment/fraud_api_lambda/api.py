import joblib
import pandas as pd
import traceback
import boto3

from fastapi import FastAPI
from mangum import Mangum
from pydantic import BaseModel, Field

# --- Configuration ---
S3_BUCKET_NAME = "aws-sam-cli-managed-default-samclisourcebucket-3ojbej2lkkdk"
S3_MODEL_KEY = "models/best_fraud_pipeline.joblib"
LOCAL_MODEL_PATH = "/tmp/best_fraud_pipeline.joblib"

# --- Model Loading ---
pipeline = None
expected_features_in = []

# This logic attempts to load the model. It's designed for the Lambda environment
# where the /tmp directory is the only writeable space.
if os.path.exists(LOCAL_MODEL_PATH):
    try:
        pipeline = joblib.load(LOCAL_MODEL_PATH)
        print("Model loaded successfully from existing /tmp path.")
    except Exception as e:
        print(f"ERROR loading model from existing /tmp file: {e}")
        traceback.print_exc()
        # pipeline remains None
else:
    try:
        print("Model not found locally. Attempting to download from S3...")
        s3 = boto3.client('s3')
        s3.download_file(S3_BUCKET_NAME, S3_MODEL_KEY, LOCAL_MODEL_PATH)
        print("Model downloaded from S3 successfully.")
        pipeline = joblib.load(LOCAL_MODEL_PATH)
        print("Model loaded successfully from newly downloaded file.")
    except Exception as e:
        print(f"ERROR downloading or loading model from S3: {e}")
        traceback.print_exc()
        # pipeline remains None

if pipeline:
    try:
        # Try to get feature names from the final step of the pipeline
        if hasattr(pipeline.steps[-1][1], 'feature_names_in_'):
            expected_features_in = list(pipeline.steps[-1][1].feature_names_in_)
            print(f"Pipeline (estimator) expects features: {expected_features_in}")
        elif hasattr(pipeline, 'feature_names_in_'):
            expected_features_in = list(pipeline.feature_names_in_)
            print(f"Pipeline (direct) expects features: {expected_features_in}")
        else:
            raise AttributeError("feature_names_in_ not found")
    except (AttributeError, IndexError):
        print("Could not automatically determine features. Using manual fallback.")
        # Provide a default list as a hard fallback
        expected_features_in = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER', 'amt_ratio_orig']
        print(f"Manually listing expected features: {expected_features_in}")
else:
    print("Pipeline failed to load, cannot determine expected features.")
    expected_features_in = []  # Ensure it's defined as an empty list on failure


class TransactionFeatures(BaseModel):
    step: int
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    type_CASH_OUT: int = Field(..., ge=0, le=1)
    type_DEBIT: int = Field(..., ge=0, le=1)
    type_PAYMENT: int = Field(..., ge=0, le=1)
    type_TRANSFER: int = Field(..., ge=0, le=1)
    amt_ratio_orig: float

    class Config:
        json_schema_extra = {
            "example": {
                "step": 10, "amount": 5000.0, "oldbalanceOrg": 20000.0, "newbalanceOrig": 15000.0,
                "oldbalanceDest": 1000.0, "newbalanceDest": 6000.0,
                "type_CASH_OUT": 1, "type_DEBIT": 0, "type_PAYMENT": 0, "type_TRANSFER": 0,
                "amt_ratio_orig": 0.25  # 5000 / 20000
            }
        }


app = FastAPI(
    title="Fraud Detection API (via SAM)",
    description="API for predicting fraudulent transactions using an XGBoost pipeline.",
    version="1.0.0"
)


@app.get("/", tags=["Status"])
async def read_root():
    """Root endpoint providing API status."""
    return {"status": "ok", "pipeline_loaded": (pipeline is not None)}


@app.post("/predict", tags=["Prediction"])
async def predict_fraud(features: TransactionFeatures):
    """Receives transaction features and returns a fraud prediction."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline is not available.")

    try:
        feature_dict = features.dict()

        if not expected_features_in:
            raise HTTPException(status_code=500, detail="Model feature expectations not loaded.")

        input_df = pd.DataFrame([feature_dict])[expected_features_in]

        prediction = pipeline.predict(input_df)
        prediction_value = int(prediction[0])

        probability_fraud = 0.0
        if hasattr(pipeline, "predict_proba"):
            probability = pipeline.predict_proba(input_df)
            probability_fraud = float(probability[0][1])

        print(f"Prediction made: Label={prediction_value}, Probability={probability_fraud:.4f}")
        return {
            "prediction_label": "Fraud" if prediction_value == 1 else "Not Fraud",
            "prediction_value": prediction_value,
            "is_fraud": bool(prediction_value),
            "probability_fraud": f"{probability_fraud:.4f}"
        }

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing expected feature in input: {e}")
    except Exception as e:
        print(f"ERROR during prediction request processing: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error during prediction: {str(e)}")


# This is the entry point for AWS Lambda, wrapping the FastAPI app with Mangum.
handler = Mangum(app)