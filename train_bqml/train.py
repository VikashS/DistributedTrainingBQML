import argparse
from google.cloud import bigquery
import json

def main(project_id, dataset_id, table_id, model_name, model_type, location):
    client = bigquery.Client(project=project_id, location=location)
    query = f"""
    CREATE OR REPLACE MODEL `{project_id}.{dataset_id}.{model_name}`
    OPTIONS(
        model_type='{model_type}',
        input_label_cols=['label']
    ) AS
    SELECT * FROM `{project_id}.{dataset_id}.{table_id}`
    """
    try:
        query_job = client.query(query)
        query_job.result()
    except Exception as e:
        print(f"Training failed for {model_name}: {e}")
        raise
    with open("/tmp/output.json", "w") as f:
        json.dump({"model_name": model_name}, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", required=True)
    parser.add_argument("--dataset_id", required=True)
    parser.add_argument("--table_id", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_type", required=True)
    parser.add_argument("--location", default="US")
    args = parser.parse_args()
    main(args.project_id, args.dataset_id, args.table_id, args.model_name, args.model_type, args.location)