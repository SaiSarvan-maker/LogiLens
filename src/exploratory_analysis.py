"""
LogiLens - Exploratory Data Analysis & Operational Bottleneck Discovery
-----------------------------------------------------------------------
Fulfills Step 3 & Step 4 of the Delivery Delay & Operations Analytics project:
1. Segments delays by warehouse, carrier, route, product type, day of week, and season.
2. Identifies high-delay combinations and bottleneck corridors.
3. Pinpoints unusual spikes and seasonal shocks (holiday peak, weather weeks).
4. Conducts statistical hypothesis tests (Chi-Square & Kruskal-Wallis).
5. Exports analytical figures and summary reports to reports/.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def load_clean_data(file_path: str = "data/processed/delivery_operations_analytics.csv") -> pd.DataFrame:
    """Loads processed operations dataset."""
    df = pd.read_csv(file_path)
    df["Shipment_Date"] = pd.to_datetime(df["Shipment_Date"])
    df["Promised_Delivery_Date"] = pd.to_datetime(df["Promised_Delivery_Date"])
    df["Actual_Delivery_Date"] = pd.to_datetime(df["Actual_Delivery_Date"])
    if "Region" in df.columns and "Origin_Region" not in df.columns:
        df["Origin_Region"] = df["Region"]
    return df


def setup_visualization_style():
    """Sets a clean, professional aesthetic for generated charts."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["figure.titlesize"] = 14


def segment_delays(df: pd.DataFrame):
    """
    Performs segmentation analysis across all 6 core dimensions:
    Warehouse, Carrier, Route, Product Type, Day of Week, and Season.
    """
    print("\n" + "=" * 65)
    print("STEP 3: OPERATIONAL DELAY SEGMENTATION")
    print("=" * 65)
    
    # 1. By Carrier
    carrier_summary = df.groupby("Carrier").agg(
        Total_Shipments=("Shipment_ID", "count"),
        Delayed_Count=("Is_Delayed", "sum"),
        On_Time_Rate=("Is_On_Time", lambda x: round(100.0 * x.mean(), 2)),
        Avg_Delay_Days=("Delay_Days", lambda x: round(x[x > 0].mean() if (x > 0).sum() > 0 else 0, 2)),
        Avg_Cost_Per_Mile=("Cost_Per_Mile", "mean")
    ).sort_values("On_Time_Rate", ascending=True)
    
    print("\n--- 1. Carrier Segmentation ---")
    print(carrier_summary.to_string())
    
    # 2. By Warehouse
    wh_summary = df.groupby(["Origin_Warehouse", "Origin_Region"]).agg(
        Total_Shipments=("Shipment_ID", "count"),
        Delayed_Count=("Is_Delayed", "sum"),
        On_Time_Rate=("Is_On_Time", lambda x: round(100.0 * x.mean(), 2)),
        Avg_Delay_Days=("Delay_Days", lambda x: round(x[x > 0].mean() if (x > 0).sum() > 0 else 0, 2))
    ).sort_values("On_Time_Rate", ascending=True)
    
    print("\n--- 2. Warehouse Segmentation ---")
    print(wh_summary.to_string())
    
    # 3. By Product Type
    prod_summary = df.groupby("Product_Type").agg(
        Total_Shipments=("Shipment_ID", "count"),
        Avg_Weight_kg=("Weight_kg", "mean"),
        On_Time_Rate=("Is_On_Time", lambda x: round(100.0 * x.mean(), 2)),
        Avg_Delay_Days=("Delay_Days", lambda x: round(x[x > 0].mean() if (x > 0).sum() > 0 else 0, 2))
    ).sort_values("On_Time_Rate", ascending=True)
    
    print("\n--- 3. Product Type Segmentation ---")
    print(prod_summary.to_string())
    
    # 4. By Day of Week
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_summary = df.groupby("Shipment_DayOfWeek").agg(
        Total_Dispatched=("Shipment_ID", "count"),
        On_Time_Rate=("Is_On_Time", lambda x: round(100.0 * x.mean(), 2)),
        Avg_Delay_Days=("Delay_Days", lambda x: round(x[x > 0].mean() if (x > 0).sum() > 0 else 0, 2))
    ).reindex(day_order)
    
    print("\n--- 4. Day of Week Dispatch Segmentation ---")
    print(day_summary.to_string())
    
    # 5. By Season & Peak Period
    season_order = ["Winter", "Spring", "Summer", "Fall"]
    season_summary = df.groupby("Season").agg(
        Total_Shipments=("Shipment_ID", "count"),
        On_Time_Rate=("Is_On_Time", lambda x: round(100.0 * x.mean(), 2)),
        Avg_Delay_Days=("Delay_Days", lambda x: round(x[x > 0].mean() if (x > 0).sum() > 0 else 0, 2)),
        Total_Concession_Cost=("Delay_Penalty_Cost", "sum")
    ).reindex(season_order)
    
    print("\n--- 5. Seasonal Performance Segmentation ---")
    print(season_summary.to_string())
    
    # 6. Top 10 Worst Routes (minimum 10 shipments)
    route_summary = df.groupby("Route").agg(
        Route_Volume=("Shipment_ID", "count"),
        Avg_Distance=("Distance_miles", "mean"),
        Delayed_Count=("Is_Delayed", "sum"),
        Delay_Rate=("Is_Delayed", lambda x: round(100.0 * x.mean(), 2)),
        Avg_Delay_Days=("Delay_Days", lambda x: round(x[x > 0].mean() if (x > 0).sum() > 0 else 0, 2))
    ).query("Route_Volume >= 10").sort_values("Delay_Rate", ascending=False).head(10)
    
    print("\n--- 6. Top 10 High-Delay Routes (Volume >= 10) ---")
    print(route_summary.to_string())
    
    return {
        "carrier": carrier_summary,
        "warehouse": wh_summary,
        "product": prod_summary,
        "day": day_summary,
        "season": season_summary,
        "route": route_summary
    }


def find_high_delay_combinations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies multi-variable interaction bottlenecks (Carrier + Warehouse, Carrier + Distance Tier).
    """
    print("\n" + "=" * 65)
    print("STEP 4: HIGH-DELAY COMBINATIONS & BOTTLENECK DISCOVERY")
    print("=" * 65)
    
    # Combination 1: Carrier x Warehouse Interaction Matrix
    pivot_comb = df.groupby(["Origin_Warehouse", "Carrier"]).agg(
        Total_Volume=("Shipment_ID", "count"),
        Delayed_Count=("Is_Delayed", "sum"),
        Delay_Rate=("Is_Delayed", lambda x: round(100.0 * x.mean(), 1))
    ).reset_index()
    
    high_delay_comb = pivot_comb[
        (pivot_comb["Total_Volume"] >= 8) & (pivot_comb["Delay_Rate"] >= 18.0)
    ].sort_values("Delay_Rate", ascending=False)
    
    print("\n--- Critical Carrier x Warehouse Bottleneck Hotspots (Delay Rate >= 18%) ---")
    print(high_delay_comb.to_string(index=False))
    
    # Combination 2: Long-Haul Carrier Fragility
    long_haul = df[df["Distance_Tier"] == "Long-Haul (>1500 mi)"].groupby("Carrier").agg(
        Volume=("Shipment_ID", "count"),
        Delayed=("Is_Delayed", "sum"),
        Delay_Rate_Pct=("Is_Delayed", lambda x: round(100.0 * x.mean(), 1)),
        Avg_Delay_Days=("Delay_Days", lambda x: round(x[x > 0].mean() if (x > 0).sum() > 0 else 0, 1))
    ).sort_values("Delay_Rate_Pct", ascending=False)
    
    print("\n--- Long-Haul (>1500 mi) Carrier Performance ---")
    print(long_haul.to_string())
    
    return high_delay_comb


def detect_unusual_spikes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies temporal anomaly weeks where delays spiked significantly above the mean.
    """
    print("\n--- Temporal Anomaly & Spike Detection ---")
    df_temp = df.copy()
    df_temp["Shipment_Week"] = df_temp["Shipment_Date"].dt.to_period("W").dt.start_time
    
    weekly = df_temp.groupby("Shipment_Week").agg(
        Weekly_Volume=("Shipment_ID", "count"),
        Delayed_Count=("Is_Delayed", "sum"),
        On_Time_Rate=("Is_On_Time", lambda x: 100.0 * x.mean()),
        Avg_Delay_Days=("Delay_Days", "mean")
    ).reset_index()
    
    # Compute z-score of weekly delay count
    weekly["Delay_ZScore"] = (weekly["Delayed_Count"] - weekly["Delayed_Count"].mean()) / weekly["Delayed_Count"].std()
    
    # Flag spikes: Z-Score > 1.8
    spikes = weekly[weekly["Delay_ZScore"] > 1.8].sort_values("Delay_ZScore", ascending=False)
    
    print(f"Detected {len(spikes)} significant weekly operational shock spikes:")
    for _, row in spikes.iterrows():
        week_str = row["Shipment_Week"].strftime("%Y-%m-%d")
        print(f"   Week of {week_str}: {int(row['Delayed_Count'])} delays ({row['Weekly_Volume']} total), OTD: {row['On_Time_Rate']:.1f}%, Z-Score: {row['Delay_ZScore']:.2f}")
        
    return spikes


def conduct_statistical_tests(df: pd.DataFrame):
    """
    Runs hypothesis tests to check if delay likelihood is statistically
    distinguishable across carriers, warehouses, and product types.
    """
    print("\n" + "=" * 65)
    print("STATISTICAL HYPOTHESIS TESTING")
    print("=" * 65)
    
    # 1. Chi-Square Test: Carrier vs Delay
    contingency_carrier = pd.crosstab(df["Carrier"], df["Is_Delayed"])
    chi2_car, p_car, dof_car, _ = stats.chi2_contingency(contingency_carrier)
    print(f"1. Chi-Square Test (Carrier vs Delay):")
    print(f"   Chi2 Statistic = {chi2_car:.3f}, p-value = {p_car:.4e} (df = {dof_car})")
    print(f"   Conclusion: {'Statistically Significant (p < 0.05)' if p_car < 0.05 else 'No Significant Evidence'}")
    
    # 2. Chi-Square Test: Warehouse vs Delay
    contingency_wh = pd.crosstab(df["Origin_Warehouse"], df["Is_Delayed"])
    chi2_wh, p_wh, dof_wh, _ = stats.chi2_contingency(contingency_wh)
    print(f"\n2. Chi-Square Test (Warehouse vs Delay):")
    print(f"   Chi2 Statistic = {chi2_wh:.3f}, p-value = {p_wh:.4e} (df = {dof_wh})")
    print(f"   Conclusion: {'Statistically Significant (p < 0.05)' if p_wh < 0.05 else 'No Significant Evidence'}")
    
    # 3. Kruskal-Wallis Test: Delay Days across Seasons
    seasons = [group["Delay_Days"].values for _, group in df.groupby("Season")]
    kw_stat, kw_p = stats.kruskal(*seasons)
    print(f"\n3. Kruskal-Wallis Test (Delay Days across Seasons):")
    print(f"   H-Statistic = {kw_stat:.3f}, p-value = {kw_p:.4e}")
    print(f"   Conclusion: {'Statistically Significant (p < 0.05)' if kw_p < 0.05 else 'No Significant Evidence'}")


def generate_and_save_visualizations(df: pd.DataFrame, output_dir: str = "reports/figures"):
    """
    Generates high-resolution visualization charts for reports and dashboard review.
    """
    os.makedirs(output_dir, exist_ok=True)
    setup_visualization_style()
    print(f"\n-> Generating visualization figures in {output_dir}...")
    
    # 1. Carrier On-Time Performance & Cost Comparison
    carrier_agg = df.groupby("Carrier").agg(
        OTD=("Is_On_Time", lambda x: 100.0 * x.mean()),
        Avg_Delay=("Delay_Days", lambda x: x[x > 0].mean() if (x > 0).sum() > 0 else 0),
        Cost_Per_Mile=("Cost_Per_Mile", "mean")
    ).sort_values("OTD", ascending=True)
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    bars = ax1.barh(carrier_agg.index, carrier_agg["OTD"], color="#2b5c8f", alpha=0.85, label="On-Time Delivery %")
    ax1.axvline(90, color="#d9534f", linestyle="--", linewidth=1.5, label="90% SLA Benchmark Target")
    ax1.set_xlabel("On-Time Delivery Rate (%)")
    ax1.set_xlim(70, 100)
    ax1.set_title("Carrier Reliability Benchmark (On-Time Delivery % vs 90% SLA)")
    
    # Annotate bar values
    for bar in bars:
        w = bar.get_width()
        ax1.text(w + 0.3, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=9, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "otd_by_carrier.png"), dpi=300)
    plt.close()
    
    # 2. Warehouse Performance Bar Chart
    wh_agg = df.groupby("Origin_Warehouse")["Is_On_Time"].mean().mul(100).sort_values(ascending=True)
    plt.figure(figsize=(10, 5))
    palette = ["#d9534f" if v < 85 else "#f0ad4e" if v < 90 else "#5cb85c" for v in wh_agg.values]
    bars = plt.barh(wh_agg.index, wh_agg.values, color=palette)
    plt.axvline(90, color="gray", linestyle="--", alpha=0.7)
    plt.xlabel("On-Time Delivery Rate (%)")
    plt.title("Origin Warehouse Dispatch & Transit Reliability")
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.3, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=9)
    plt.xlim(75, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "otd_by_warehouse.png"), dpi=300)
    plt.close()
    
    # 3. Carrier x Warehouse Heatmap
    pivot_heatmap = df.pivot_table(
        index="Origin_Warehouse", 
        columns="Carrier", 
        values="Is_Delayed", 
        aggfunc=lambda x: round(100.0 * x.mean(), 1)
    )
    plt.figure(figsize=(11, 7))
    sns.heatmap(pivot_heatmap, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={'label': 'Delay Rate (%)'})
    plt.title("Operational Hotspot Matrix: Delay Rate (%) by Warehouse & Carrier")
    plt.xlabel("Carrier")
    plt.ylabel("Origin Warehouse")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "carrier_warehouse_heatmap.png"), dpi=300)
    plt.close()
    
    # 4. Day of Week & Seasonality Comparison
    day_season = df.pivot_table(
        index="Shipment_DayOfWeek", 
        columns="Season", 
        values="Is_Delayed", 
        aggfunc=lambda x: round(100.0 * x.mean(), 1)
    ).reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])[["Winter", "Spring", "Summer", "Fall"]]
    
    plt.figure(figsize=(9, 5))
    sns.heatmap(day_season, annot=True, fmt=".1f", cmap="Blues", cbar_kws={'label': 'Delay Rate (%)'})
    plt.title("Seasonal & Day-of-Week Delay Rate Heatmap (%)")
    plt.xlabel("Season")
    plt.ylabel("Shipment Day of Week")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "delay_by_day_and_season.png"), dpi=300)
    plt.close()
    
    print("   Visualizations saved successfully.")


def run_eda():
    """Main EDA pipeline."""
    df = load_clean_data()
    segments = segment_delays(df)
    high_delay = find_high_delay_combinations(df)
    spikes = detect_unusual_spikes(df)
    conduct_statistical_tests(df)
    generate_and_save_visualizations(df)
    
    # Save bottleneck summary table
    os.makedirs("reports", exist_ok=True)
    high_delay.to_csv("reports/operational_bottlenecks_summary.csv", index=False)
    print("\n-> Saved bottleneck summary to reports/operational_bottlenecks_summary.csv")


if __name__ == "__main__":
    run_eda()
