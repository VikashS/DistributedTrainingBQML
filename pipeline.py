
from kfp import dsl
from kfp.dsl import component, Output, Dataset, Model, Metrics
from google_cloud_pipeline_components.v1.bigquery import BigqueryExportModelJobOp

# ETL Component
@component(
    base_image="gcr.io/your-project-id/etl-component:latest",
    output_component_file="etl_component.yaml",
)
def etl_component(
    project_id: str,
    dataset_id: str,
    source_table: str,
    output_table: str,
    location: str = "US",
    output_table_path: Output[Dataset],
):
    pass

# Training Component (single model)
@component(
    base_image="gcr.io/your-project-id/train-bqml-component:latest",
    output_component_file="train_component.yaml",
)
def train_component(
    project_id: str,
    dataset_id: str,
    table_id: str,
    model_name: str,
    model_type: str,
    location: str = "US",
    model_output: Output[Model],
):
    pass

# Evaluate Models Component
@component(
    base_image="gcr.io/your-project-id/evaluate-models-component:latest",
    output_component_file="evaluate_component.yaml",
)
def evaluate_models(
    project_id: str,
    dataset_id: str,
    model_names: str,
    location: str = "US",
    best_model: Output[Model],
    best_metric: Output[ Kocetrics],
):
    pass

@dsl.pipeline(
    name="bqml-dockerized-parallel-pipeline",
    description="Pipeline with dockerized ETL, parallel BigQuery ML training, and evaluation",
)
def bqml_pipeline(
    project_id: str,
    dataset_id: str,
    source_table: str,
    output_table: str,
    location: str = "US",
    pipeline_root: str = "gs://your-bucket/pipeline_root",
):
    # Step 1: ETL Component
    etl_task = etl_component(
        project_id=project_id,
        dataset_id=dataset_id,
        source_table=source_table,
        output_table=output_table,
        location=location
    )

    # Step 2: Train multiple models in parallel
    model_configs = [
        {"name": "logistic_model", "type": "LOGISTIC_REG"},
        {"name": "boosted_model", "type": "BOOSTED_TREE_CLASSIFIER"},
        {"name": "dnn_model", "type": "DNN_CLASSIFIER"},
    ]

    training_tasks = []
    for config in model_configs:
        task = train_component(
            project_id=project_id,
            dataset_id=dataset_id,
            table_id=output_table,
            model_name=config["name"],
            model_type=config["type"],
            location=location
        ).after(etl_task)
        training_tasks.append(task)

    # Step 3: Evaluate models
    eval_task = evaluate_models(
        project_id=project_id,
        dataset_id=dataset_id,
        model_names=",".join([config["name"] for config in model_configs]),
        location=location
    ).after(*training_tasks)  # Waits for all training tasks

    # Step 4: Export the best model
    export_task = BigqueryExportModelJobOp(
        project=project_id,
        location=location,
        model=f"{project_id}.{dataset_id}.{eval_task.outputs['best_model'].metadata['resourceName'].split('/')[-1]}",
        model_destination_path=f"gs://your-bucket/models/{eval_task.outputs['best_model'].metadata['resourceName'].split('/')[-1]}"
    ).after(eval_task)