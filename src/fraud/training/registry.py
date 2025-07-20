from .algorithms.random_forest import RandomForestModel
from .algorithms.xgboost import XGBoostModel
from .algorithms.isolation_forest import IsolationForestModel
from .algorithms.ensemble import EnsembleModel

MODEL_REGISTRY = {
    "random_forest": RandomForestModel,
    "xgboost": XGBoostModel,
    "isolation_forest": IsolationForestModel,
    "ensemble": EnsembleModel
}

def get_model(model_name: str, **kwargs):
    """Factory function to get model instance"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_name}")
    return MODEL_REGISTRY[model_name](**kwargs)