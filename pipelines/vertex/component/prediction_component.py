from kfp.v2.dsl import component, Input, Output, Dataset


@component(
    base_image="python:3.11",
    output_component_file="prediction_component.yaml"
)
def prediction_component(
        model: Input[Model],
        input_data: Input[Dataset],
        predictions: Output[Dataset],
        batch_size: int = 1000
):
    """Component for batch predictions using trained model"""
    import joblib
    import pandas as pd
    from fraud.prediction.main import predict

    # Load model and make predictions
    predictions_df = predict(
        model_path=model.path,
        input_path=input_data.path,
        batch_size=batch_size
    )

    # Save predictions
    predictions_df.to_parquet(predictions.path)

    print(f"Predictions saved to {predictions.path}")