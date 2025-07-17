# fraud/etl/data_preparation.py
import argparse
import logging
from fraud.utils import bq_utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_etl(project_id: str, dataset_id: str, source_table: str, processed_table: str):
    """
    Performs ETL operations by executing a BigQuery SQL query.
    1. Reads raw fraud transaction data from BigQuery.
    2. Performs feature engineering (example: transaction amount, time-based features).
    3. Stores processed data in a new BigQuery table.
    """
    logger.info(f"Starting ETL process for BigQuery table: {project_id}.{dataset_id}.{source_table}")

    etl_query = f"""
    CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{processed_table}` AS
    SELECT
        transaction_id,
        user_id,
        amount,
        EXTRACT(HOUR FROM timestamp) AS transaction_hour,
        CAST(EXTRACT(DAYOFWEEK FROM timestamp) IN (1, 7) AS INT64) AS is_weekend,
        -- Add more feature engineering here (e.g., aggregation, categorical encoding)
        is_fraud
    FROM
        `{project_id}.{dataset_id}.{source_table}`
    WHERE
        timestamp IS NOT NULL
    """

    bq_utils.execute_bq_query(project_id, etl_query)
    logger.info(f"ETL completed. Processed data now in: {project_id}.{dataset_id}.{processed_table}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ETL process for fraud data using BigQuery ML.")
    parser.add_argument("--project_id", type=str, required=True, help="Google Cloud project ID.")
    parser.add_argument("--dataset_id", type=str, required=True, help="BigQuery Dataset ID.")
    parser.add_argument("--source_table", type=str, required=True, help="Source BigQuery table name (raw data).")
    parser.add_argument("--processed_table", type=str, required=True, help="Output BigQuery processed table name.")
    args = parser.parse_args()

    run_etl(args.project_id, args.dataset_id, args.source_table, args.processed_table)