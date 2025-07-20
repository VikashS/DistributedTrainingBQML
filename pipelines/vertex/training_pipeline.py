from google.cloud import aiplatform
from kfp.v2 import dsl
from kfp.v2.dsl import (
    pipeline,
    Artifact,
    Dataset,
    Input,
    Model,
    Output
)
import os
from typing import Dict, List

# Import components
from .components.etl_component import etl_component
from .components.training_component import training_component
from .components.prediction_component import prediction_component


def create_custom_training_job(
        model_type: str,
        serving_image_uri: str = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"
) -> dsl.ContainerOp:
    """Factory for creating CustomContainerTrainingJob components"""

    @dsl.component(
        base_image=f"us-docker.pkg.dev/{os.getenv('GOOGLE_CLOUD_PROJECT')}/training-{model_type}:latest",
        output_component_file=f"components/{model_type}_trainer.yaml"
    )
    def custom_train_component(
            input_data: Input[Dataset],
            model_output: Output[Model],
            config_path: str = f"/app/configs/models/{model_type}.yaml"
    ):
        """Wrapper for CustomContainerTrainingJob"""
        job = aiplatform.CustomContainerTrainingJob(
            display_name=f"train-{model_type}",
            container_uri=f"us-docker.pkg.dev/{os.getenv('GOOGLE_CLOUD_PROJECT')}/training-{model_type}:latest",
            command=["python", "-m", "fraud.training.main"],
            args=[
                "--input-data", input_data.path,
                "--model-output", model_output.path,
                "--model-type", model_type,
                "--config-path", config_path
            ],
            staging_bucket=f"gs://{os.getenv('GOOGLE_CLOUD_PROJECT')}-staging"
        )

        # Configure resources based on model type
        resource_config = {
            "machine_type": "n1-standard-8" if model_type == "xgboost" else "n1-standard-4",
            "accelerator_type": "NVIDIA_TESLA_T4" if model_type == "xgboost" else None,
            "accelerator_count": 1 if model_type == "xgboost" else None
        }

        job.run(
            model_display_name=f"fraud-{model_type}",
            model_serving_container_image_uri=serving_image_uri,
            **resource_config
        )

    return custom_train_component


@pipeline(
    name="fraud-detection-parallel-training",
    pipeline_root=os.getenv("PIPELINE_ROOT", "gs://your-bucket/pipeline-root")
)
def training_pipeline(
        input_path: str = "gs://your-bucket/input/data.csv",
        first_stage_models: List[str] = ["random_forest", "xgboost", "isolation_forest"],
        ensemble_config: Dict[str, Any] = None
):
    """Main pipeline with parallel training and ensemble modeling"""

    # ETL Stage
    etl_task = etl_component(
        input_path=input_path
    )

    # First Stage: Parallel Model Training
    first_stage_tasks = {}

    with dsl.ParallelFor(first_stage_models) as model_type:
        # Create model-specific training component
        trainer = create_custom_training_job(model_type)

        # Execute training job
        first_stage_tasks[model_type] = trainer(
            input_data=etl_task.outputs["output_data"]
        )

    # Second Stage: Ensemble Training
    ensemble_trainer = create_custom_training_job("ensemble")(
        input_data=etl_task.outputs["output_data"]
    )

    # Final Prediction
    predictor = prediction_component(
        model=ensemble_trainer.outputs["model_output"],
        input_data=etl_task.outputs["output_data"]
    )