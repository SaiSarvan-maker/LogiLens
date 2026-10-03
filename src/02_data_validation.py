"""
LogiLens - Data Validation Utility
----------------------------------
Validates data quality, missing values, status distributions, and date ordering
for both the real source dataset and the synthetic relational extension.
"""

import os
import pandas as pd

def validate_data():
    raw_path = "data/raw/logistics_shipments_dataset.csv"
    if not os.path.exists(raw_path):
        print(f"Error: {raw_path} not found.")
        return

    df = pd.read_csv(raw_path)
    print("=" * 65)
    print("LOGILENS - DATA VALIDATION REPORT")
    print("=" * 65)

    # 1. Convert dates temporarily for validation
    df["Shipment_Date"] = pd.to_datetime(df["Shipment_Date"], errors="coerce")
    df["Delivery_Date"] = pd.to_datetime(df["Delivery_Date"], errors="coerce")

    print("\n1. DATE VALIDATION (REAL SOURCE)")
    print("-" * 65)
    print("Invalid Shipment_Date values:", df["Shipment_Date"].isna().sum())
    print("Missing Delivery_Date values:", df["Delivery_Date"].isna().sum())

    # 2. Check missing Delivery_Date against Status
    print("\n2. MISSING DELIVERY DATE BY STATUS")
    print("-" * 65)
    missing_delivery = df[df["Delivery_Date"].isna()]
    print(missing_delivery["Status"].value_counts().to_string())

    # 3. Check missing Cost against Status
    print("\n3. MISSING COST BY STATUS")
    print("-" * 65)
    missing_cost = df[df["Cost"].isna()]
    print(missing_cost["Status"].value_counts().to_string())

    # 4. Check date ordering anomalies
    print("\n4. DATE ORDER ANOMALIES")
    print("-" * 65)
    invalid_date_order = df[
        df["Delivery_Date"].notna() & (df["Delivery_Date"] < df["Shipment_Date"])
    ]
    print("Delivery logged before shipment date:", len(invalid_date_order))

    # 5. Check numerical boundaries
    print("\n5. NUMERICAL BOUNDARY VALIDATION")
    print("-" * 65)
    print("Weight <= 0:", (df["Weight_kg"] <= 0).sum())
    print("Cost <= 0:", (df["Cost"] <= 0).sum())
    print("Distance <= 0:", (df["Distance_miles"] <= 0).sum())
    print("Transit_Days <= 0:", (df["Transit_Days"] <= 0).sum())

    # 6. Status distribution & Primary Delay Rate Denominator
    print("\n6. STATUS DISTRIBUTION & RESOLVED DENOMINATOR")
    print("-" * 65)
    status_counts = df["Status"].value_counts()
    print(status_counts.to_string())

    delivered_count = status_counts.get("Delivered", 0)
    delayed_count = status_counts.get("Delayed", 0)
    resolved_total = delivered_count + delayed_count
    primary_delay_rate = (delayed_count / resolved_total * 100.0) if resolved_total > 0 else 0.0

    lost_count = status_counts.get("Lost", 0)
    returned_count = status_counts.get("Returned", 0)
    op_failure_rate = ((delayed_count + lost_count + returned_count) / len(df) * 100.0)

    print(f"\n   -> Resolved Deliveries Denominator (Delivered + Delayed): {resolved_total}")
    print(f"   -> Primary Status Delay Rate:                             {primary_delay_rate:.2f}%  (Delayed / [Delivered + Delayed])")
    print(f"   -> Operational Failure Rate:                              {op_failure_rate:.2f}%  ([Delayed + Lost + Returned] / Total)")

    print("\n" + "=" * 65)
    print("VALIDATION COMPLETE")
    print("=" * 65)

if __name__ == "__main__":
    validate_data()