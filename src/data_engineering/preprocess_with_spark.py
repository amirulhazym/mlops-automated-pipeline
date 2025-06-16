import logging
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, log1p

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def setup_arg_parser():
    """Sets up the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Spark Preprocessing Script for Fraud Detection Data"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input raw CSV file (e.g., data/raw/full_fraud_data.csv).",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        required=True,
        help="Suffix for the output directory (e.g., 'full' or 'sample_1M').",
    )
    return parser


def main(args):
    """Main function to run the Spark preprocessing job."""
    logging.info(
        f"Starting Spark preprocessing job for data version: {args.output_suffix}"
    )

    # --- Dynamic Path Construction --- (updated)
    raw_data_path = args.input_file
    processed_data_path = f"data/processed/engineered_features_{args.output_suffix}"

    spark = None
    try:
        # 1. Create SparkSession
        spark = (
            SparkSession.builder.appName(f"FraudDataPreprocessing_{args.output_suffix}")
            .master("local[*]")
            .getOrCreate()
        )
        logging.info("SparkSession created.")

        # 2. Load Raw Data
        raw_df = spark.read.csv(raw_data_path, header=True, inferSchema=True)
        logging.info(
            f"Raw data loaded from {raw_data_path}. Row count: {raw_df.count()}"
        )

        # 3. Perform Data Cleaning & Feature Engineering (based on EDA findings from P2L0)

        # Drop unnecessary columns identified in EDA
        processed_df = raw_df.drop("nameOrig", "nameDest", "isFlaggedFraud")

        # Apply log transformation to skewed numerical features
        skewed_cols = [
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
        ]
        for c in skewed_cols:
            # Use log1p which computes log(1+x) to handle zero values gracefully
            processed_df = processed_df.withColumn(f"{c}_log", log1p(col(c)))

        # One-hot encode the 'type' column
        # This creates new binary columns for each transaction type
        types = [
            row["type"] for row in processed_df.select("type").distinct().collect()
        ]
        for t in types:
            processed_df = processed_df.withColumn(
                f"type_{t}", when(col("type") == t, 1).otherwise(0)
            )

        # Drop original columns that have been transformed
        # We keep 'isFraud' which is our target.
        cols_to_drop = skewed_cols + ["type"]
        final_df = processed_df.drop(*cols_to_drop)

        logging.info("Feature engineering complete.")
        final_df.printSchema()

        # 4. Save Processed Data as Parquet
        final_df.write.mode("overwrite").parquet(processed_data_path)
        logging.info(
            f"Processed data saved to {processed_data_path} in Parquet format."
        )

    except Exception as e:
        logging.error(f"Error during Spark preprocessing: {e}", exc_info=True)
        raise
    finally:
        if spark:
            spark.stop()
            logging.info("SparkSession stopped.")

    logging.info("Spark preprocessing job finished successfully.")


if __name__ == "__main__":
    arg_parser = setup_arg_parser()
    cli_args = arg_parser.parse_args()
    main(cli_args)
