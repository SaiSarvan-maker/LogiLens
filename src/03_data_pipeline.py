"""
LogiLens - Data Pipeline & Operational Metrics Engineering
----------------------------------------------------------
Fulfills Step 1 & Step 2 of the Delivery Delay & Operations Analytics project:
1. Combines order, shipment, warehouse, carrier, and delivery-event data.
2. Cleans data anomalies (swapped dates, missing values, non-positive weights).
3. Computes Promised Delivery Date, Actual Delivery Date, Delay Days, and On-Time flags.
4. Enriches with operational segments: Route, Day of Week, Season, Distance Tier.
5. Saves clean analytical tables to CSV, Parquet, and SQLite.
"""

import os
import sqlite3
import pandas as pd
import numpy as np


def load_raw_sources(data_dir: str = "data/raw"):
    """Load all normalized relational sources and master raw extract."""
    print("-> Loading relational data sources...")
    df_orders = pd.read_csv(os.path.join(data_dir, "orders.csv"))
    df_shipments = pd.read_csv(os.path.join(data_dir, "shipments.csv"))
    df_warehouses = pd.read_csv(os.path.join(data_dir, "warehouses.csv"))
    df_carriers = pd.read_csv(os.path.join(data_dir, "carriers.csv"))
    df_events = pd.read_csv(os.path.join(data_dir, "delivery_events.csv"))
    
    return df_orders, df_shipments, df_warehouses, df_carriers, df_events


def clean_and_merge_data(df_orders, df_shipments, df_warehouses, df_carriers, df_events):
    """
    Combines the relational tables and resolves real-world data quality issues.
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
    merged["Promised_Delivery_Date"] = pd.to_datetime(merged["Promised_Delivery_Date"], errors="coerce")
    merged["Actual_Delivery_Date"] = pd.to_datetime(merged["Actual_Delivery_Date"], errors="coerce")
    
    # 3. Data Cleaning: Inverted Dates
    # Detect instances where Actual_Delivery_Date < Shipment_Date (date entry inversion anomaly)
    inverted_mask = (
        merged["Actual_Delivery_Date"].notna() &
        (merged["Actual_Delivery_Date"] < merged["Shipment_Date"])
    )
    inverted_count = inverted_mask.sum()
    if inverted_count > 0:
        print(f"   Fixing {inverted_count} date inversion anomalies (Actual Delivery < Shipment Date)...")
        # In reality, this is usually caused by day/month typo; correct to Shipment Date + Promised SLA + delta
        merged.loc[inverted_mask, "Actual_Delivery_Date"] = (
            merged.loc[inverted_mask, "Shipment_Date"] + 
            pd.to_timedelta(merged.loc[inverted_mask, "Promised_SLA_Days"], unit="D")
        )
    
    # 4. Data Cleaning: Missing Delivery Date for Completed/Delayed Orders
    missing_deliv_mask = (
        merged["Actual_Delivery_Date"].isna() & 
        merged["Delivery_Status"].isin(["Delivered", "Delayed"])
    )
    missing_deliv_count = missing_deliv_mask.sum()
    if missing_deliv_count > 0:
        print(f"   Imputing {missing_deliv_count} missing delivery dates for closed orders...")
        # If marked Delayed, actual delivery is promised + 2 days; if Delivered, promised date
        extra_days = np.where(merged.loc[missing_deliv_mask, "Delivery_Status"] == "Delayed", 2, 0)
        merged.loc[missing_deliv_mask, "Actual_Delivery_Date"] = (
            merged.loc[missing_deliv_mask, "Promised_Delivery_Date"] + 
            pd.to_timedelta(extra_days, unit="D")
        )
    
    # 5. Data Cleaning: Invalid Weight (Weight <= 0)
    invalid_weight_mask = (merged["Weight_kg"] <= 0)
    if invalid_weight_mask.sum() > 0:
        print(f"   Correcting {invalid_weight_mask.sum()} invalid weight values (Weight <= 0)...")
        # Impute with median weight of that product type
        median_weights = merged.groupby("Product_Type")["Weight_kg"].transform("median")
        merged.loc[invalid_weight_mask, "Weight_kg"] = median_weights.loc[invalid_weight_mask]
        # Safeguard fallback
        merged["Weight_kg"] = merged["Weight_kg"].replace(0, 15.0)

    # 6. Data Cleaning: Missing Cost (41 records)
    missing_cost_mask = merged["Cost"].isna()
    if missing_cost_mask.sum() > 0:
        print(f"   Imputing {missing_cost_mask.sum()} missing shipping cost values based on carrier rates & distance...")
        # Cost estimate: Base_Rate_Per_Mile * Distance_miles * (1 + Weight / 100)
        imputed_cost = (
            merged["Base_Rate_Per_Mile"] * 
            merged["Distance_miles"] * 
            (1.0 + (merged["Weight_kg"] / 100.0))
        ).round(2)
        merged.loc[missing_cost_mask, "Cost"] = imputed_cost.loc[missing_cost_mask]

    return merged


def compute_operational_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all operational metrics, delay KPIs, and business segmentation features.
    """
    print("-> Computing operational delay metrics and business segments...")
    df = df.copy()
    
    # 1. Core Delay Metrics
    # Days variance: Actual - Promised (positive = late, negative = early)
    df["Days_Variance"] = (df["Actual_Delivery_Date"] - df["Promised_Delivery_Date"]).dt.days
    
    # Delay days: Non-negative days delayed (0 if on-time or early)
    df["Delay_Days"] = np.where(df["Days_Variance"] > 0, df["Days_Variance"], 0)
    # Fill NaN for non-delivered shipments (Lost/In Transit)
    df["Delay_Days"] = df["Delay_Days"].fillna(0).astype(int)
    
    # Is_Delayed flag: 1 if status is Delayed or Days_Variance > 0
    df["Is_Delayed"] = np.where(
        (df["Delivery_Status"] == "Delayed") | (df["Days_Variance"] > 0),
        1,
        0
    )
    
    # Lost or Returned orders are also operational failures
    df["Is_Operational_Failure"] = np.where(
        (df["Is_Delayed"] == 1) | (df["Delivery_Status"].isin(["Lost", "Returned"])),
        1,
        0
    )
    
    # On-Time Delivery Flag
    df["Is_On_Time"] = np.where(
        (df["Delivery_Status"] == "Delivered") & (df["Is_Delayed"] == 0),
        1,
        0
    )
    
    # Delay Severity Tier
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

    # 2. Routing & Geographical Segmentation
    df["Origin_Warehouse"] = df["Warehouse_Name"]
    df["Route"] = df["City"] + " -> " + df["Destination_City"]
    
    def categorize_distance(miles):
        if miles < 500:
            return "Short-Haul (<500 mi)"
        elif miles <= 1500:
            return "Medium-Haul (500-1500 mi)"
        else:
            return "Long-Haul (>1500 mi)"
            
    df["Distance_Tier"] = df["Distance_miles"].apply(categorize_distance)
    
    # 3. Temporal & Seasonality Segmentation
    df["Shipment_DayOfWeek"] = df["Shipment_Date"].dt.day_name()
    df["Shipment_DayOfWeek_Num"] = df["Shipment_Date"].dt.dayofweek  # 0=Monday, 6=Sunday
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
    
    # 4. Financial & Cost Efficiency Metrics
    df["Cost_Per_Mile"] = (df["Cost"] / df["Distance_miles"]).round(3)
    df["Cost_Per_Kg"] = (df["Cost"] / df["Weight_kg"]).round(2)
    
    # Estimated delay penalty cost (SLA concessions, expedited support, customer credits)
    # Average industry benchmark: 20% of shipping cost + $15 administrative cost per delayed package
    df["Delay_Penalty_Cost"] = np.where(
        df["Is_Delayed"] == 1,
        (df["Cost"] * 0.20 + 15.0).round(2),
        0.0
    )
    
    # Rename columns for clarity in analytics
    df["Carrier"] = df["Carrier_Name"]
    df["Origin_City"] = df["City"]
    df["Origin_State"] = df["State"]
    df["Destination"] = df["Destination_City"]
    df["Status"] = df["Delivery_Status"]

    return df


def save_analytics_dataset(df: pd.DataFrame, output_dir: str = "data/processed", db_path: str = "data/logistics.db"):
    """
    Saves clean processed data to CSV, Parquet, and an indexed SQLite database.
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
    
    # Store processed analytical table
    df.to_sql("delivery_operations_analytics", conn, if_exists="replace", index=False)
    
    # Create indexes
    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_carrier ON delivery_operations_analytics(Carrier)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_warehouse ON delivery_operations_analytics(Origin_Warehouse)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_is_delayed ON delivery_operations_analytics(Is_Delayed)")
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
    
    # Display Summary KPIs
    total_volume = len(analytics_df)
    delivered_volume = len(analytics_df[analytics_df["Status"].isin(["Delivered", "Delayed"])])
    on_time_count = analytics_df["Is_On_Time"].sum()
    delayed_count = analytics_df["Is_Delayed"].sum()
    otp = (on_time_count / delivered_volume) * 100
    avg_delay = analytics_df[analytics_df["Is_Delayed"] == 1]["Delay_Days"].mean()
    total_penalty = analytics_df["Delay_Penalty_Cost"].sum()
    
    print("\n" + "=" * 65)
    print("DATA PIPELINE EXECUTION SUMMARY")
    print("=" * 65)
    print(f"Total Shipments Ingested:        {total_volume:,}")
    print(f"Delivered / Resolved Volume:     {delivered_volume:,}")
    print(f"On-Time Deliveries:              {on_time_count:,}")
    print(f"Delayed Shipments:               {delayed_count:,}")
    print(f"On-Time Delivery Rate (OTD%):    {otp:.2f}%")
    print(f"Average Delay (Delayed Orders):  {avg_delay:.2f} days")
    print(f"Estimated Delay Concession Cost: ${total_penalty:,.2f}")
    print("=" * 65)


if __name__ == "__main__":
    run_pipeline()
