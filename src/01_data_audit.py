"""
LogiLens - Data Audit Utility
-----------------------------
Audits both the Real Source Extract (data/raw/logistics_shipments_dataset.csv)
and the Synthetic Relational Extension (data/synthetic/*.csv).
"""

import os
import pandas as pd

def audit_data():
    raw_path = "data/raw/logistics_shipments_dataset.csv"
    print("=" * 65)
    print("LOGILENS - DATA AUDIT & PROVENANCE REPORT")
    print("=" * 65)
    
    if os.path.exists(raw_path):
        print("\n1. REAL SOURCE EXTRACT (data/raw/logistics_shipments_dataset.csv)")
        print("-" * 65)
        df_raw = pd.read_csv(raw_path)
        print("Shape:", df_raw.shape)
        print("Columns:", df_raw.columns.tolist())
        print("\nData types:")
        print(df_raw.dtypes)
        print("\nMissing values:")
        print(df_raw.isna().sum())
        print("\nDuplicate rows:", df_raw.duplicated().sum())
        print("\nStatus Distribution (Observed Ground Truth):")
        print(df_raw["Status"].value_counts())
    else:
        print(f"Error: {raw_path} not found.")

    synthetic_dir = "data/synthetic"
    if os.path.exists(synthetic_dir):
        print("\n2. SYNTHETIC RELATIONAL EXTENSION (data/synthetic/*.csv)")
        print("-" * 65)
        for fname in ["orders.csv", "shipments.csv", "warehouses.csv", "carriers.csv", "delivery_events.csv"]:
            fpath = os.path.join(synthetic_dir, fname)
            if os.path.exists(fpath):
                s_df = pd.read_csv(fpath)
                print(f"  • {fname:22s}: {s_df.shape[0]} rows, {s_df.shape[1]} cols")
    else:
        print(f"\nSynthetic directory ({synthetic_dir}) not found. Run src/generate_relational_data.py first.")

if __name__ == "__main__":
    audit_data()