import argparse
import json
import os
from typing import Dict, Any
from .registry import get_model
import joblib


def train_model(
        input_data: str,
        model_output: str,
        model_type: str,
        config_path: str = None,
        hyperparameters: Dict[str, Any] = None
):
    """Main training function executed in container"""
    # Load config if provided
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        hyperparameters = config.get('hyperparameters', {})

    # Initialize model
    model = get_model(model_type, **hyperparameters)

    # Load training data
    train_data = joblib.load(input_data)

    # Train model
    model.fit(train_data['X'], train_data['y'])

    # Save model
    os.makedirs(os.path.dirname(model_output), exist_ok=True)
    joblib.dump(model, model_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", required=True)
    parser.add_argument("--model-output", required=True)
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--config-path")
    args = parser.parse_args()

    train_model(
        input_data=args.input_data,
        model_output=args.model_output,
        model_type=args.model_type,
        config_path=args.config_path
    )