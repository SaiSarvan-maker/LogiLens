"""
LogiLens - Data Pipeline & Operational Metrics Engineering
----------------------------------------------------------
Alias module delegating to 03_data_pipeline.py
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))

from src import data_pipeline
from src import generate_relational_data
import importlib

# Import all public symbols from 03_data_pipeline
_impl = importlib.import_module("src.03_data_pipeline")

load_raw_sources = _impl.load_raw_sources
clean_and_merge_data = _impl.clean_and_merge_data
compute_operational_kpis = _impl.compute_operational_kpis
save_analytics_dataset = _impl.save_analytics_dataset
run_pipeline = _impl.run_pipeline

if __name__ == "__main__":
    run_pipeline()
