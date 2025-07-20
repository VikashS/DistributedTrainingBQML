from kfp.v2.dsl import component, Output, Dataset
from google.cloud import aiplatform


@component(
    base_image="python:3.11",
    output_component_file="etl_component.yaml"
)
def etl_component(
        input_path: str,
        output_data: Output[Dataset],
        config_path: str = "/app/configs/etl_config.yaml"
):
    """ETL component that processes raw data into training-ready format"""
    import yaml
    import pandas as pd
    from fraud.etl.main import process_data

    # Initialize Vertex AI
    aiplatform.init(
        project=aiplatform.gapic.JobServiceClient().project_path(
            aiplatform.gapic.JobServiceClient().project
        ),
        location=aiplatform.gapic.JobServiceClient().location
    )

    # Process data using your fraud package
    process_data(
        input_path=input_path,
        output_path=output_data.path,
        config_path=config_path
    )

    print(f"ETL completed. Output saved to {output_data.path}")