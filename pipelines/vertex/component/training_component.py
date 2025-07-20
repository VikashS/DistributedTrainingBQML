from kfp.v2.dsl import component, Input, Output, Model
from typing import Dict, Any


@component(
    base_image="python:3.11",
    output_component_file="training_component.yaml"
)
def training_component(
        input_data: Input[Dataset],
        model_output: Output[Model],
        model_type: str,
        hyperparameters: Dict[str, Any] = None,
        config_path: str = None
):
    """Custom training component using Vertex AI's CustomContainerTrainingJob"""
    from google.cloud import aiplatform
    from fraud.training.registry import get_model
    import json
    import os

    # Initialize Vertex AI
    aiplatform.init(
        project=os.getenv("GOOGLE_CLOUD_PROJECT"),
        location=os.getenv("GOOGLE_CLOUD_REGION")
    )

    # Load hyperparameters from config if not provided
    if hyperparameters is None and config_path:
        with open(config_path) as f:
            hyperparameters = json.load(f).get("hyperparameters", {})

    # Initialize and train model
    model = get_model(model_type, **hyperparameters)
    model.fit(input_data.path)

    # Save trained model
    model.save(model_output.path)

    # Register model in Vertex AI Model Registry
    aiplatform.Model.upload(
        display_name=f"fraud-{model_type}",
        artifact_uri=os.path.dirname(model_output.path),
        serving_container_image_uri=os.getenv("SERVING_IMAGE_URI")
    )

    print(f"Training completed for {model_type}. Model saved to {model_output.path}")