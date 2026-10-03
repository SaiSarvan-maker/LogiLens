"""
LogiLens - Data Pipeline & Operational Metrics Engineering
----------------------------------------------------------
Fulfills Step 1 & Step 2 of the Delivery Delay & Operations Analytics project:
1. Combines real raw extract and synthetic relational sources with explicit provenance.
2. Cleans anomalies (inverted dates, invalid weights) and preserves missingness flags (Delivery_Date_Missing, Cost_Imputed).
3. Computes Primary Status-Based Delay metrics (Option B) and Analytical SLA Baseline (Option A).
4. Calculates Operational Failure Rate and Estimated Concession Exposure.
5. Saves clean analytical tables to CSV, Parquet, and SQLite.
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np

# Ensure project root directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.generate_relational_data import generate_relational_tables


def load_raw_sources(raw_dir: str = "data/raw", synthetic_dir: str = "data/synthetic"):
    """Load real raw extract and synthetic relational sources."""
    print("-> Loading relational data sources...")
    
    # Ensure synthetic extension files exist
    if not os.path.exists(os.path.join(synthetic_dir, "orders.csv")):
        print("   Synthetic extension tables missing. Generating synthetic relational structure...")
        generate_relational_tables(raw_dir=raw_dir, synthetic_dir=synthetic_dir)
        
    df_orders = pd.read_csv(os.path.join(synthetic_dir, "orders.csv"))
    df_shipments = pd.read_csv(os.path.join(synthetic_dir, "shipments.csv"))
    df_warehouses = pd.read_csv(os.path.join(synthetic_dir, "warehouses.csv"))
    df_carriers = pd.read_csv(os.path.join(synthetic_dir, "carriers.csv"))
    df_events = pd.read_csv(os.path.join(synthetic_dir, "delivery_events.csv"))
    
    return df_orders, df_shipments, df_warehouses, df_carriers, df_events


def clean_and_merge_data(df_orders, df_shipments, df_warehouses, df_carriers, df_events):
    """
    Combines the relational tables and resolves data quality issues with honest lineage.
    """
    print("-> Merging relational sources into unified operational fact table...")
    
    # 1. Relational Merges
    merged = df_shipments.merge(df_orders, on="Order_ID", how="inner")
    merged = merged.merge(df_warehouses, on="Warehouse_ID", how="inner")
    merged = merged.merge(df_carriers, on="Carrier_ID", how="inner")
    merged = merged.merge(df_events, on="Shipment_ID", how="inner")
    
    print(f"   Successfully joined {len(merged)} operational records.")
    
    # 2. Date parsing and validation
    merged["Shipment_Date"] = pd.to_datetime(merged["Shipment_Date"], errors="coerce")
    
    # SLA Date: Renamed & documented as Derived_Promised_Delivery_Date (Analytical expected SLA)
    if "Derived_Promised_Delivery_Date" in merged.columns:
        merged["Derived_Promised_Delivery_Date"] = pd.to_datetime(merged["Derived_Promised_Delivery_Date"], errors="coerce")
    elif "Promised_Delivery_Date" in merged.columns:
        merged["Derived_Promised_Delivery_Date"] = pd.to_datetime(merged["Promised_Delivery_Date"], errors="coerce")
    else:
        merged["Derived_Promised_Delivery_Date"] = merged["Shipment_Date"] + pd.to_timedelta(merged["Promised_SLA_Days"], unit="D")
        
    merged["Actual_Delivery_Date"] = pd.to_datetime(merged["Actual_Delivery_Date"], errors="coerce")
    
    # 3. Data Cleaning: Inverted Dates (Actual Delivery < Shipment Date)
    inverted_mask = (
        merged["Actual_Delivery_Date"].notna() &
        (merged["Actual_Delivery_Date"] < merged["Shipment_Date"])
    )
    inverted_count = inverted_mask.sum()
    if inverted_count > 0:
        print(f"   Flagging {inverted_count} date inversion anomalies (Actual Delivery < Shipment Date)...")
        # Inverted delivery dates are set to NaT to prevent synthetic bias
        merged.loc[inverted_mask, "Actual_Delivery_Date"] = pd.NaT
    
    # 4. Data Quality Flag: Missing Delivery Date (DO NOT IMPUTE for primary status analysis)
    # Flag preserves missingness info without circular date creation
    merged["Delivery_Date_Missing"] = np.where(merged["Actual_Delivery_Date"].isna(), 1, 0)
    missing_deliv_cnt = merged["Delivery_Date_Missing"].sum()
    if missing_deliv_cnt > 0:
        print(f"   Flagged {missing_deliv_cnt} records with missing delivery dates (Delivery_Date_Missing = 1).")

    # 5. Data Cleaning: Invalid Weight (Weight <= 0)
    invalid_weight_mask = (merged["Weight_kg"] <= 0)
    if invalid_weight_mask.sum() > 0:
        print(f"   Correcting {invalid_weight_mask.sum()} invalid weight values (Weight <= 0)...")
        median_weights = merged.groupby("Product_Type")["Weight_kg"].transform("median")
        merged.loc[invalid_weight_mask, "Weight_kg"] = median_weights.loc[invalid_weight_mask]
        merged["Weight_kg"] = merged["Weight_kg"].replace(0, 15.0)

    # 6. Data Quality & Imputation: Missing Cost (41 records)
    missing_cost_mask = merged["Cost"].isna()
    merged["Cost_Imputed"] = np.where(missing_cost_mask, 1, 0)
    if missing_cost_mask.sum() > 0:
        print(f"   Estimating {missing_cost_mask.sum()} missing shipping cost values (Cost_Imputed = 1)...")
        imputed_cost = (
            merged["Base_Rate_Per_Mile"] * 
            merged["Distance_miles"] * 
            (1.0 + (merged["Weight_kg"] / 100.0))
        ).round(2)
        merged.loc[missing_cost_mask, "Cost"] = imputed_cost.loc[missing_cost_mask]

    return merged


def compute_operational_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes primary status metrics (Option B), analytical SLA metrics (Option A),
    and operational failure categories.
    """
    print("-> Computing operational delay metrics and business segments...")
    df = df.copy()
    
    # -------------------------------------------------------------
    # 1. Primary Target & Status Definitions (Option B - Primary Analysis)
    # -------------------------------------------------------------
    # Is_Delayed_Status: 1 if Delivery_Status == "Delayed", else 0
    df["Is_Delayed_Status"] = np.where(df["Delivery_Status"] == "Delayed", 1, 0)
    df["Is_Delayed"] = df["Is_Delayed_Status"]  # Primary delay flag alias
    
    # Resolved shipment flag: Delivered or Delayed (Denominator for Delay Rate)
    df["Is_Resolved"] = np.where(df["Delivery_Status"].isin(["Delivered", "Delayed"]), 1, 0)
    
    # On-Time flag for primary analysis
    df["Is_On_Time"] = np.where(df["Delivery_Status"] == "Delivered", 1, 0)
    
    # Operational Failure Flag: Delayed, Lost, or Returned
    df["Is_Operational_Failure"] = np.where(
        df["Delivery_Status"].isin(["Delayed", "Lost", "Returned"]), 1, 0
    )
    
    # Operational Category breakdown
    def categorize_outcome(status):
        if status == "Delivered":
            return "Delivered (On-Time)"
        elif status == "Delayed":
            return "Delayed (Late)"
        elif status == "Lost":
            return "Lost (Shipment Failure)"
        elif status == "Returned":
            return "Returned (Delivery Failure)"
        else:
            return "In Transit (Unresolved)"
            
    df["Operational_Outcome_Category"] = df["Delivery_Status"].apply(categorize_outcome)
    
    # -------------------------------------------------------------
    # 2. Analytical SLA Baseline Metrics (Option A - Secondary Extension)
    # -------------------------------------------------------------
    df["Promised_Delivery_Date"] = df["Derived_Promised_Delivery_Date"]
    
    # Days variance against derived SLA: Actual - Derived Promised Date
    df["Days_Variance"] = np.where(
        df["Actual_Delivery_Date"].notna(),
        (df["Actual_Delivery_Date"] - df["Derived_Promised_Delivery_Date"]).dt.days,
        np.nan
    )
    
    # Delay days: Non-negative variance against derived SLA
    df["Delay_Days"] = np.where(
        df["Days_Variance"].notna() & (df["Days_Variance"] > 0),
        df["Days_Variance"],
        0
    )
    df["Delay_Days"] = df["Delay_Days"].fillna(0).astype(int)
    
    def categorize_delay(days):
        if days <= 0:
            return "On-Time / Early"
        elif days <= 2:
            return "Minor Delay (1-2 Days)"
        elif days <= 4:
            return "Moderate Delay (3-4 Days)"
        else:
            return "Severe Delay (5+ Days)"
            
    df["Delay_Severity"] = df["Delay_Days"].apply(categorize_delay)

    # -------------------------------------------------------------
    # 3. Routing & Geographic Segments
    # -------------------------------------------------------------
    df["Origin_Warehouse"] = df["Warehouse_Name"]
    df["Carrier"] = df["Carrier_Name"]
    df["Origin_City"] = df["City"]
    df["Origin_State"] = df["State"]
    df["Destination"] = df["Destination_City"]
    df["Status"] = df["Delivery_Status"]
    df["Route"] = df["City"] + " -> " + df["Destination_City"]
    
    def categorize_distance(miles):
        if miles < 500:
            return "Short-Haul (<500 mi)"
        elif miles <= 1500:
            return "Medium-Haul (500-1500 mi)"
        else:
            return "Long-Haul (>1500 mi)"
            
    df["Distance_Tier"] = df["Distance_miles"].apply(categorize_distance)
    
    # -------------------------------------------------------------
    # 4. Temporal & Seasonality Segments
    # -------------------------------------------------------------
    df["Shipment_DayOfWeek"] = df["Shipment_Date"].dt.day_name()
    df["Shipment_DayOfWeek_Num"] = df["Shipment_Date"].dt.dayofweek
    df["Shipment_Month"] = df["Shipment_Date"].dt.month_name()
    df["Shipment_Month_Num"] = df["Shipment_Date"].dt.month
    df["Shipment_Quarter"] = "Q" + df["Shipment_Date"].dt.quarter.astype(str)
    
    def get_season(month):
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Fall"
            
    df["Season"] = df["Shipment_Month_Num"].apply(get_season)
    df["Is_Peak_Season"] = np.where(df["Shipment_Month_Num"].isin([11, 12]), 1, 0)
    
    # -------------------------------------------------------------
    # 5. Financial & Exposure Metrics
    # -------------------------------------------------------------
    df["Cost_Per_Mile"] = (df["Cost"] / df["Distance_miles"]).round(3)
    df["Cost_Per_Kg"] = (df["Cost"] / df["Weight_kg"]).round(2)
    
    # Scenario estimate based on assumed concession formula: 20% freight cost + $15 admin fee
    df["Estimated_Concession_Exposure"] = np.where(
        df["Delivery_Status"] == "Delayed",
        (df["Cost"] * 0.20 + 15.0).round(2),
        0.0
    )
    df["Delay_Penalty_Cost"] = df["Estimated_Concession_Exposure"]  # Backward compatibility alias

    return df


def save_analytics_dataset(df: pd.DataFrame, output_dir: str = "data/processed", db_path: str = "data/logistics.db"):
    """
    Saves clean processed data to CSV, Parquet, and SQLite.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "delivery_operations_analytics.csv")
    parquet_path = os.path.join(output_dir, "delivery_operations_analytics.parquet")
    
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    print(f"-> Saved processed CSV: {csv_path} ({len(df)} rows)")
    print(f"-> Saved processed Parquet: {parquet_path}")
    
    # Save to SQLite database
    print(f"-> Exporting tables to SQLite database at {db_path}...")
    conn = sqlite3.connect(db_path)
    
    df.to_sql("delivery_operations_analytics", conn, if_exists="replace", index=False)
    
    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_carrier ON delivery_operations_analytics(Carrier)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_warehouse ON delivery_operations_analytics(Origin_Warehouse)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON delivery_operations_analytics(Status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_route ON delivery_operations_analytics(Route)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_season ON delivery_operations_analytics(Season)")
    conn.commit()
    conn.close()
    print("-> SQLite database export and indexing complete.")


def run_pipeline():
    """Main execution orchestrator."""
    print("=" * 65)
    print("LOGILENS - DELIVERY OPERATIONS DATA PIPELINE (ETL)")
    print("=" * 65)
    
    df_orders, df_shipments, df_warehouses, df_carriers, df_events = load_raw_sources()
    merged = clean_and_merge_data(df_orders, df_shipments, df_warehouses, df_carriers, df_events)
    analytics_df = compute_operational_kpis(merged)
    save_analytics_dataset(analytics_df)
    
    # Display Summary KPIs (Defensible Status-Based Calculations)
    total_volume = len(analytics_df)
    resolved_df = analytics_df[analytics_df["Status"].isin(["Delivered", "Delayed"])]
    resolved_volume = len(resolved_df)
    on_time_count = len(analytics_df[analytics_df["Status"] == "Delivered"])
    delayed_count = len(analytics_df[analytics_df["Status"] == "Delayed"])
    lost_count = len(analytics_df[analytics_df["Status"] == "Lost"])
    returned_count = len(analytics_df[analytics_df["Status"] == "Returned"])
    in_transit_count = len(analytics_df[analytics_df["Status"] == "In Transit"])
    
    primary_delay_rate = (delayed_count / resolved_volume * 100.0) if resolved_volume > 0 else 0.0
    op_failure_rate = ((delayed_count + lost_count + returned_count) / total_volume * 100.0)
    total_exposure = analytics_df["Estimated_Concession_Exposure"].sum()
    imputed_costs = analytics_df["Cost_Imputed"].sum()
    
    print("\n" + "=" * 65)
    print("DATA PIPELINE EXECUTION SUMMARY")
    print("=" * 65)
    print(f"Total Network Shipments:             {total_volume:,}")
    print(f"  • Delivered (On-Time):             {on_time_count:,}")
    print(f"  • Delayed (Late):                  {delayed_count:,}")
    print(f"  • Lost (Shipment Failure):          {lost_count:,}")
    print(f"  • Returned (Delivery Failure):      {returned_count:,}")
    print(f"  • In Transit (Unresolved):          {in_transit_count:,}")
    print("-" * 65)
    print(f"Resolved Deliveries Denominator:     {resolved_volume:,}")
    print(f"Primary Status Delay Rate:           {primary_delay_rate:.2f}%  (Delayed / [Delivered + Delayed])")
    print(f"Operational Failure Rate:            {op_failure_rate:.2f}%  ([Delayed + Lost + Returned] / Total)")
    print(f"Estimated Concession Exposure:       ${total_exposure:,.2f}  (Scenario Estimate)")
    print(f"Cost Imputation Flags (Cost_Imputed):{imputed_costs} records")
    print("=" * 65)


if __name__ == "__main__":
    run_pipeline()

