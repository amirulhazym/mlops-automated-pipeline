# scripts/kafka_producer.py
import time
import json
import random
from kafka import KafkaProducer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KAFKA_TOPIC = "fraud_transactions_topic"
KAFKA_BROKER_URL = "localhost:9092" # As advertised in docker-compose.yml

def create_producer():
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BROKER_URL,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        logger.info("KafkaProducer created successfully.")
        return producer
    except Exception as e:
        logger.error(f"Error creating KafkaProducer: {e}")
        return None

def generate_dummy_transaction():
    # Mimic features expected by your P1 API
    # Ensure these match the Pydantic model of your P1 API
    return {
        "step": random.randint(1, 744),
        "amount": round(random.uniform(1.0, 10000.0), 2),
        "oldbalanceOrg": round(random.uniform(0.0, 50000.0), 2),
        "newbalanceOrig": round(random.uniform(0.0, 50000.0), 2),
        "oldbalanceDest": round(random.uniform(0.0, 50000.0), 2),
        "newbalanceDest": round(random.uniform(0.0, 50000.0), 2),
        "type_CASH_OUT": random.choice([0, 1]), # Simple one-hot example
        "type_DEBIT": random.choice([0, 1]),
        "type_PAYMENT": random.choice([0, 1]),
        "type_TRANSFER": random.choice([0, 1]),
        "amt_ratio_orig": round(random.uniform(0.0, 1.0), 3)
    }

if __name__ == "__main__":
    producer = create_producer()
    if producer:
        try:
            for i in range(10): # Send 10 dummy transactions
                transaction = generate_dummy_transaction()
                logger.info(f"Sending transaction {i+1}: {transaction}")
                producer.send(KAFKA_TOPIC, value=transaction)
                producer.flush() # Ensure message is sent
                time.sleep(random.uniform(0.5, 2.0)) # Simulate varying arrival times
            logger.info("Finished sending all transactions.")
        except Exception as e:
            logger.error(f"Error during message sending: {e}")
        finally:
            producer.close()
            logger.info("KafkaProducer closed.")
    else:
        logger.error("Could not start producer.")
