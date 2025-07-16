import argparse
from google.cloud import bigquery

def main(project_id, dataset_id, source_table, output_table, location):
    client = bigquery.Client(project=project_id, location=location)
    etl_query = f"""
    CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{output_table}` AS
    SELECT
        IFNULL(numeric_feature, 0) AS numeric_feature,
        CASE
            WHEN categorical_feature IS NULL THEN 'unknown'
            ELSE categorical_feature
        END AS categorical_feature,
        label
    FROM `{project_id}.{dataset_id}.{source_table}`
    WHERE label IS NOT NULL
    """
    query_job = client.query(etl_query)
    query_job.result()
    with open("/tmp/output_table.txt", "w") as f:
        f.write(f"{project_id}.{dataset_id}.{output_table}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", required=True)
    parser.add_argument("--dataset_id", required=True)
    parser.add_argument("--source_table", required=True)
    parser.add_argument("--output_table", required=True)
    parser.add_argument("--location", default="US")
    args = parser.parse_args()
    main(args.project_id, args.dataset_id, args.source_table, args.output_table, args.location)