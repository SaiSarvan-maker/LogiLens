"""
LogiLens - Machine Learning Delay Risk Estimator
------------------------------------------------
Alias module delegating to 05_delay_risk_model.py
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))

import importlib

_impl = importlib.import_module("src.05_delay_risk_model")

load_modeling_data = _impl.load_modeling_data
build_preprocessor = _impl.build_preprocessor
train_and_evaluate_models = _impl.train_and_evaluate_models
plot_model_diagnostics = _impl.plot_model_diagnostics
save_model_artifacts = _impl.save_model_artifacts
predict_delay_risk = _impl.predict_delay_risk
run_training_pipeline = _impl.run_training_pipeline

if __name__ == "__main__":
    run_training_pipeline()
