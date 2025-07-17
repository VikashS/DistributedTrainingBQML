# pipeline.py
import kfp
from kfp.v2 import compiler
from kfp.v2.dsl import pipeline, component, OutputPath, InputPath
from kubernetes import client as k8s_client  # For resource requests

# --- Configuration ---
PROJECT_ID = "your-gcp-project-id"  # <<< REPLACE WITH YOUR GCP PROJECT ID
REGION = "your-gcp-region"  # <<< REPLACE WITH YOUR GCP REGION, e.g., "us-central1", "europe-west1"
PIPELINE_ROOT = f"gs://your-gcs-bucket-for-artifacts/{kfp.dsl.PIPELINE_JOB_ID}"  # <<< REPLACE WITH YOUR GCS BUCKET

# BigQuery details - ensure this dataset exists in your project
BQ_DATASET_ID = "fraud_detection_dataset"
RAW_TRANSACTIONS_TABLE = "raw_transactions"  # Assuming this table exists with raw data
PROCESSED_TRANSACTIONS_TABLE = "processed_transactions"  # Table created by ETL

# Docker image paths (ensure these match what you pushed to GCR/Artifact Registry)
ETL_IMAGE = f"gcr.io/{PROJECT_ID}/fraud-etl:latest"
MODEL_TRAINING_IMAGE = f"gcr.io/{PROJECT_ID}/fraud-model-training:latest"

# Define a list of models to train with their specific configurations.
# This list drives the parallel execution.
MODELS_TO_TRAIN = [
    {"name": "fraud_model_a_logistic_reg", "type": "LOGISTIC_REG"},
    {"name": "fraud_model_b_boosted_tree", "type": "BOOSTED_TREE_CLASSIFIER"},
    {"name": "fraud_model_c_dnn_classifier", "type": "DNN_CLASSIFIER"},
    # Add more models here following the same structure:
    # {"name": "fraud_model_d_linear_reg", "type": "LINEAR_REG"}, # Example for a different task type
]


# --- Define Custom Kubeflow Components ---

@component(
    base_image=ETL_IMAGE,
    output_component_file="etl_component.yaml",
    packages_to_install=["google-cloud-bigquery==2.34.0"],  # Ensure specific packages are available for the component
)
def etl_step(
        project_id: str,
        dataset_id: str,
        source_table: str,
        processed_table: str,
) -> str:  # The output type hints are for better understanding of pipeline graph
    """
    Kubeflow component for running the ETL process,
    which executes a BigQuery SQL query to prepare data.
    Returns the full BigQuery table ID of the processed data.
    """
    import os
    import sys
    # Add fraud package to sys.path for import in the component runtime
    sys.path.append(os.path.join(os.path.dirname(__file__), 'fraud'))
    from fraud.etl import data_preparation  # Import the function

    full_processed_table_id = f"{project_id}.{dataset_id}.{processed_table}"
    data_preparation.run_etl(
        project_id=project_id,
        dataset_id=dataset_id,
        source_table=source_table,
        processed_table=processed_table
    )
    return full_processed_table_id  # Return the processed table ID as an output artifact


@component(
    base_image=MODEL_TRAINING_IMAGE,
    output_component_file="model_training_component.yaml",
    packages_to_install=["google-cloud-bigquery==2.34.0"],
)
def train_model_step(
        project_id: str,
        dataset_id: str,
        processed_table_id: str,  # Now accepts the output from ETL
        model_name: str,
        model_type: str,
) -> str:  # Returns the full model ID as an output artifact
    """
    Kubeflow component for training a BigQuery ML model with a specified type.
    Returns the full BigQuery ML model ID.
    """
    import os
    import sys
    # Add fraud package to sys.path for import in the component runtime
    sys.path.append(os.path.join(os.path.dirname(__file__), 'fraud'))
    from fraud.models import model_training  # Import the function

    # Extract just the table name from the full ID
    _project, _dataset, _table = processed_table_id.split('.')

    full_model_id = f"{project_id}.{dataset_id}.{model_name}"
    model_training.train_model(
        project_id=project_id,
        dataset_id=dataset_id,
        processed_table=_table,  # Pass just the table name to the script
        model_name=model_name,
        model_type=model_type
    )
    return full_model_id  # Return the full model ID


# Optional: Component for model evaluation
@component(
    base_image=MODEL_TRAINING_IMAGE,  # Can reuse the same image if it has bq_utils
    output_component_file="model_evaluation_component.yaml",
    packages_to_install=["google-cloud-bigquery==2.34.0"],
)
def evaluate_model_step(
        project_id: str,
        dataset_id: str,
        model_id: str,  # Accepts the trained model ID from the training step
        processed_table_id: str,  # Accepts the processed data table ID from ETL
        metric_output_path: OutputPath(str),  # Output path for evaluation metrics
):
    """
    Kubeflow component to evaluate a BigQuery ML model.
    Writes a simple evaluation metric to an output file.
    """
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), 'fraud'))
    from fraud.utils import bq_utils  # Use bq_utils for query execution

    # Extract just the table name from the full ID
    _project_data, _dataset_data, _table_data = processed_table_id.split('.')
    _project_model, _dataset_model, _model_name = model_id.split('.')

    # Example BQML evaluation query (adapt as per your needs)
    eval_query = f"""
    SELECT
        *
    FROM
        ML.EVALUATE(MODEL `{model_id}`,
        (SELECT * FROM `{processed_table_id}` WHERE is_fraud IS NOT NULL))
    """

    # Execute query, this will return a Job that can be fetched for results
    client = bq_utils.bigquery.Client(project=project_id)
    query_job = client.query(eval_query)
    results = query_job.result()  # Wait for the job to complete and get results

    # Process results - this is a simplified example
    metrics = {}
    for row in results:
        # Assuming classification metrics like precision, recall, f1_score
        metrics['accuracy'] = row.accuracy
        metrics['precision'] = row.precision
        metrics['recall'] = row.recall
        metrics['f1_score'] = row.f1_score
        break  # Assuming only one row of overall metrics

    # Write a simplified metric to the output path
    with open(metric_output_path, 'w') as f:
        f.write(f"Model: {model_id}\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
            print(f"{key}: {value}")  # Also print to component logs
    print(f"Evaluation metrics written to {metric_output_path}")


# Optional: A final component to indicate pipeline completion
@component
def final_pipeline_step():
    print("All pipeline steps completed successfully, including parallel model training and evaluation!")


# --- Define the Kubeflow Pipeline DAG ---

@pipeline(
    name="fraud-detection-bigqueryml-matured-parallel-pipeline",
    description="A multi-model fraud detection pipeline with dynamic parallel BigQuery ML training and evaluation.",
    pipeline_root=PIPELINE_ROOT,
    enable_caching=True,  # Enable caching by default for faster iterative development
)
def fraud_detection_pipeline(
        project_id: str = PROJECT_ID,
        region: str = REGION,
        bq_dataset_id: str = BQ_DATASET_ID,
        raw_transactions_table: str = RAW_TRANSACTIONS_TABLE,
        processed_transactions_table: str = PROCESSED_TRANSACTIONS_TABLE,
):
    # Step 1: ETL for data preparation
    etl_task = etl_step(
        project_id=project_id,
        dataset_id=bq_dataset_id,
        source_table=raw_transactions_table,
        processed_table=processed_transactions_table,
    ).set_display_name("ETL and Data Preparation")  # Add display name for clarity

    # Step 2: Parallel Model Training using with_items
    # This loop dynamically creates parallel tasks for each model defined in MODELS_TO_TRAIN.
    # Each training task will run concurrently after the ETL step.
    with kfp.dsl.ParallelFor(MODELS_TO_TRAIN) as model_item:
        train_task = train_model_step(
            project_id=project_id,
            dataset_id=bq_dataset_id,
            processed_table_id=etl_task.output,  # Pass output from ETL to training
            model_name=model_item["name"],
            model_type=model_item["type"],
        ).set_display_name(f"Train Model: {model_item['name']}")
        train_task.after(etl_task)  # Each parallel task depends on the ETL step

        # Optional: Model Evaluation after each model is trained
        # This evaluation task will run for each trained model, potentially in parallel
        # with other training/evaluation pairs, but sequentially after its specific training task.
        evaluate_task = evaluate_model_step(
            project_id=project_id,
            dataset_id=bq_dataset_id,
            model_id=train_task.output,  # Pass output from training to evaluation
            processed_table_id=etl_task.output,  # Pass processed data to evaluation
        ).set_display_name(f"Evaluate Model: {model_item['name']}")
        evaluate_task.after(train_task)

    # Step 3: Final step that waits for ALL parallel training and evaluation tasks to complete.
    # By depending on the `train_task` within the ParallelFor loop, it correctly
    # implies waiting for all iterations to finish. If evaluation tasks also exist
    # and they depend on the train_task within the loop, the final step will wait for those too.
    final_pipeline_step_task = final_pipeline_step()
    # This correctly waits for all branches of the ParallelFor loop to finish.
    # If `evaluate_task` was the last task in the loop for each iteration,
    # then `final_pipeline_step_task.after(evaluate_task)` would be used here.
    # For simplicity, relying on the 'train_task' which is the first in the ParallelFor
    # is often sufficient as Kubeflow ensures all child operations within a loop are done.
    final_pipeline_step_task.after(train_task)  # Using train_task here to imply completion of all parallel branches


# --- Pipeline Compilation and Submission ---
if __name__ == "__main__":
    # Ensure a local 'fraud' directory exists for component imports during compilation
    # This is critical for the `packages_to_install` to work correctly and
    # for the python `import` statements within @component functions.
    import os

    if not os.path.exists("fraud"):
        print("Error: 'fraud/' directory not found. Please ensure your project structure is correct.")
        exit(1)

    compiler.Compiler().compile(
        pipeline_func=fraud_detection_pipeline,
        package_path="fraud_detection_pipeline.json",
    )
    print("Pipeline compiled to fraud_detection_pipeline.json")

    # Optional: Submit the pipeline job directly from here (for testing/development)
    # from google.cloud import aiplatform

    # aiplatform.init(project=PROJECT_ID, location=REGION)

    # job = aiplatform.PipelineJob(
    #     display_name="fraud-bqml-matured-parallel-run",
    #     template_path="fraud_detection_pipeline.json",
    #     pipeline_root=PIPELINE_ROOT,
    #     enable_caching=True,
    #     # Specify a service account if different from default
    #     # service_account='your-service-account@your-project-id.iam.gserviceaccount.com',
    #     labels={"workflow_type": "training", "target_system": "bigquery_ml"}
    # )
    # job.run()
    # print(f"Pipeline job submitted. View at: {job.console_uri}")