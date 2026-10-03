"""
LogiLens - Exploratory Data Analysis & Operational Bottleneck Discovery
-----------------------------------------------------------------------
Alias module delegating to 04_exploratory_analysis.py
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))

import importlib

_impl = importlib.import_module("src.04_exploratory_analysis")

load_clean_data = _impl.load_clean_data
segment_delays = _impl.segment_delays
find_high_delay_combinations = _impl.find_high_delay_combinations
detect_unusual_spikes = _impl.detect_unusual_spikes
conduct_statistical_tests = _impl.conduct_statistical_tests
generate_and_save_visualizations = _impl.generate_and_save_visualizations
run_eda = _impl.run_eda

if __name__ == "__main__":
    run_eda()
