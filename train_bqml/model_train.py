# fraud/models/model_training.py
import argparse
import logging
from fraud.utils import bq_utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(project_id: str, dataset_id: str, processed_table: str, model_name: str, model_type: str):
    """
    Trains a BigQuery ML model based on the specified model_type.
    """
    logger.info(f"Starting training for model: '{model_name}' (Type: {model_type}) in {project_id}.{dataset_id}")

    train_query = ""
    # Define feature columns common to all models in this example
    feature_cols = "amount, transaction_hour, is_weekend"

    if model_type.upper() == 'LOGISTIC_REG':
        train_query = f"""
        CREATE OR REPLACE MODEL `{project_id}.{dataset_id}.{model_name}`
        OPTIONS(
            MODEL_TYPE='LOGISTIC_REG',
            INPUT_LABEL_COLS=['is_fraud'],
            AUTO_CLASS_WEIGHTS=TRUE,
            MAX_ITERATIONS=10 # Example: Added a hyperparameter
        ) AS
        SELECT
            {feature_cols},
            is_fraud
        FROM
            `{project_id}.{dataset_id}.{processed_table}`
        WHERE
            is_fraud IS NOT NULL
        """
    elif model_type.upper() == 'BOOSTED_TREE_CLASSIFIER':
        train_query = f"""
        CREATE OR REPLACE MODEL `{project_id}.{dataset_id}.{model_name}`
        OPTIONS(
            MODEL_TYPE='BOOSTED_TREE_CLASSIFIER',
            INPUT_LABEL_COLS=['is_fraud'],
            BOOSTER_TYPE='DART',
            NUM_PARALLEL_TREE=10,
            MAX_TREE_DEPTH=8,
            L1_REG=0.1 # Example: Added a hyperparameter
        ) AS
        SELECT
            {feature_cols},
            is_fraud
        FROM
            `{project_id}.{dataset_id}.{processed_table}`
        WHERE
            is_fraud IS NOT NULL
        """
    elif model_type.upper() == 'DNN_CLASSIFIER':
        train_query = f"""
        CREATE OR REPLACE MODEL `{project_id}.{dataset_id}.{model_name}`
        OPTIONS(
            MODEL_TYPE='DNN_CLASSIFIER',
            INPUT_LABEL_COLS=['is_fraud'],
            HIDDEN_UNITS=[32, 16],
            OPTIMIZER='ADAM',
            DROPOUT=[0.2, 0.2] # Example: Added a hyperparameter
        ) AS
        SELECT
            {feature_cols},
            is_fraud
        FROM
            `{project_id}.{dataset_id}.{processed_table}`
        WHERE
            is_fraud IS NOT NULL
        """
    else:
        logger.error(f"Unsupported model_type: {model_type}. Please check supported BQML model types.")
        raise ValueError(f"Unsupported model_type: {model_type}")

    bq_utils.execute_bq_query(project_id, train_query)
    logger.info(f"Model '{model_name}' (Type: {model_type}) training completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BigQuery ML model.")
    parser.add_argument("--project_id", type=str, required=True, help="Google Cloud project ID.")
    parser.add_argument("--dataset_id", type=str, required=True, help="BigQuery Dataset ID.")
    parser.add_argument("--processed_table", type=str, required=True, help="Processed BigQuery table name.")
    parser.add_argument("--model_name", type=str, required=True, help="Name for the trained BQML model.")
    parser.add_argument("--model_type", type=str, required=True, help="Type of BQML model (e.g., LOGISTIC_REG, BOOSTED_TREE_CLASSIFIER, DNN_CLASSIFIER).")
    args = parser.parse_args()

    train_model(args.project_id, args.dataset_id, args.processed_table, args.model_name, args.model_type)