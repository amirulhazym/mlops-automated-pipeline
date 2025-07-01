# scripts/kafka_consumer_api_caller.py
import json
from kafka import KafkaConsumer
import requests # To call P1 API
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
KAFKA_TOPIC = "fraud_transactions_topic"
KAFKA_BROKER_URL = "localhost:9092"
KAFKA_CONSUMER_GROUP = "fraud_api_callers"

# API Endpoint: Choose one to target
# Option 1: Lambda API (if deployed and we want to test against it)
# API_ENDPOINT = "YOUR_LAMBDA_P1_API_ENDPOINT/predict" 
# Option 2: Minikube Service (get URL via `minikube service p1-fraud-api-service --url`)
API_ENDPOINT = "http://127.0.0.1:8080/predict" # get this by "minikube service p1-fraud-api-service --profile mlops-luster"

def create_consumer():
    try:
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=KAFKA_BROKER_URL,
            auto_offset_reset='earliest', # Start reading at the earliest message if new consumer
            enable_auto_commit=True,
            group_id=KAFKA_CONSUMER_GROUP,
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        logger.info("KafkaConsumer created successfully.")
        return consumer
    except Exception as e:
        logger.error(f"Error creating KafkaConsumer: {e}")
        return None

if __name__ == "__main__":
    consumer = create_consumer()
    if consumer:
        logger.info(f"Listening for messages on topic '{KAFKA_TOPIC}'...")
        try:
            for message in consumer:
                transaction_data = message.value
                logger.info(f"Received transaction: {transaction_data}")
                
                try:
                    # Call P1 API (Lambda or Minikube service)
                    response = requests.post(API_ENDPOINT, json=transaction_data, timeout=10)
                    response.raise_for_status() # Raise an exception for HTTP errors
                    prediction_result = response.json()
                    logger.info(f"API Prediction for transaction: {prediction_result}")
                except requests.exceptions.RequestException as e_req:
                    logger.error(f"API call failed for transaction: {transaction_data}. Error: {e_req}")
                except Exception as e_api:
                    logger.error(f"Error processing API response for {transaction_data}. Error: {e_api}")

        except KeyboardInterrupt:
            logger.info("Consumer interrupted. Closing...")
        except Exception as e:
            logger.error(f"Error in consumer loop: {e}")
        finally:
            consumer.close()
            logger.info("KafkaConsumer closed.")
    elif not consumer:
        logger.error("Could not start Kafka consumer.")
    else:
        logger.error(f"API Endpoint not configured correctly: {API_ENDPOINT}. Exiting.")
        