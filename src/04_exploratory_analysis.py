"""
LogiLens - Exploratory Data Analysis & Operational Bottleneck Discovery
-----------------------------------------------------------------------
Fulfills Step 3 & Step 4 of the Delivery Delay & Operations Analytics project:
1. Re-framed Narrative: "Observed patterns and operational factors associated with delayed shipment status."
2. Evaluates Primary Status Delay Rate (Delayed / [Delivered + Delayed]) and Operational Failure Rate across 6 dimensions.
3. Identifies multi-variable interaction bottlenecks and temporal anomaly shock spikes.
4. Conducts statistical hypothesis tests (Chi-Square for categorical association & Kruskal-Wallis for transit days).
5. Exports figures and summary tables to reports/.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Ensure root directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def load_clean_data(file_path: str = "data/processed/delivery_operations_analytics.csv") -> pd.DataFrame:
    """Loads processed operations dataset."""
    df = pd.read_csv(file_path)
    df["Shipment_Date"] = pd.to_datetime(df["Shipment_Date"])
    df["Derived_Promised_Delivery_Date"] = pd.to_datetime(df.get("Derived_Promised_Delivery_Date", df.get("Promised_Delivery_Date")))
    df["Actual_Delivery_Date"] = pd.to_datetime(df["Actual_Delivery_Date"], errors="coerce")
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
    Performs segmentation analysis across 6 core dimensions using the primary status-based
    delay rate (Delayed / [Delivered + Delayed]) and Operational Failure Rate.
    """
    print("\n" + "=" * 65)
    print("STEP 3: OBSERVED DELAY PATTERNS & OPERATIONAL SEGMENTATION")
    print("=" * 65)
    print("NOTE: Primary Delay Rate is computed strictly on resolved shipments (Delivered + Delayed).")
    
    # Filter resolved shipments
    resolved = df[df["Status"].isin(["Delivered", "Delayed"])].copy()
    
    # 1. By Carrier
    carrier_summary = df.groupby("Carrier").agg(
        Total_Shipments=("Shipment_ID", "count"),
        Resolved_Shipments=("Status", lambda x: x.isin(["Delivered", "Delayed"]).sum()),
        Delayed_Count=("Status", lambda x: (x == "Delayed").sum()),
        Primary_Delay_Rate=("Status", lambda x: round(100.0 * (x == "Delayed").sum() / x.isin(["Delivered", "Delayed"]).sum(), 2) if x.isin(["Delivered", "Delayed"]).sum() > 0 else 0.0),
        Operational_Failure_Rate=("Status", lambda x: round(100.0 * x.isin(["Delayed", "Lost", "Returned"]).sum() / len(x), 2)),
        Avg_Cost_Per_Mile=("Cost_Per_Mile", "mean")
    ).sort_values("Primary_Delay_Rate", ascending=False)
    
    print("\n--- 1. Carrier Segmentation ---")
    print(carrier_summary.to_string())
    
    # 2. By Warehouse
    wh_summary = df.groupby(["Origin_Warehouse", "Origin_Region"]).agg(
        Total_Shipments=("Shipment_ID", "count"),
        Resolved_Shipments=("Status", lambda x: x.isin(["Delivered", "Delayed"]).sum()),
        Delayed_Count=("Status", lambda x: (x == "Delayed").sum()),
        Primary_Delay_Rate=("Status", lambda x: round(100.0 * (x == "Delayed").sum() / x.isin(["Delivered", "Delayed"]).sum(), 2) if x.isin(["Delivered", "Delayed"]).sum() > 0 else 0.0),
        Operational_Failure_Rate=("Status", lambda x: round(100.0 * x.isin(["Delayed", "Lost", "Returned"]).sum() / len(x), 2))
    ).sort_values("Primary_Delay_Rate", ascending=False)
    
    print("\n--- 2. Warehouse Segmentation ---")
    print(wh_summary.to_string())
    
    # 3. By Product Type
    prod_summary = df.groupby("Product_Type").agg(
        Total_Shipments=("Shipment_ID", "count"),
        Avg_Weight_kg=("Weight_kg", "mean"),
        Delayed_Count=("Status", lambda x: (x == "Delayed").sum()),
        Primary_Delay_Rate=("Status", lambda x: round(100.0 * (x == "Delayed").sum() / x.isin(["Delivered", "Delayed"]).sum(), 2) if x.isin(["Delivered", "Delayed"]).sum() > 0 else 0.0),
        Operational_Failure_Rate=("Status", lambda x: round(100.0 * x.isin(["Delayed", "Lost", "Returned"]).sum() / len(x), 2))
    ).sort_values("Primary_Delay_Rate", ascending=False)
    
    print("\n--- 3. Product Type Segmentation ---")
    print(prod_summary.to_string())
    
    # 4. By Day of Week
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_summary = df.groupby("Shipment_DayOfWeek").agg(
        Total_Dispatched=("Shipment_ID", "count"),
        Delayed_Count=("Status", lambda x: (x == "Delayed").sum()),
        Primary_Delay_Rate=("Status", lambda x: round(100.0 * (x == "Delayed").sum() / x.isin(["Delivered", "Delayed"]).sum(), 2) if x.isin(["Delivered", "Delayed"]).sum() > 0 else 0.0),
        Operational_Failure_Rate=("Status", lambda x: round(100.0 * x.isin(["Delayed", "Lost", "Returned"]).sum() / len(x), 2))
    ).reindex(day_order)
    
    print("\n--- 4. Day of Week Dispatch Segmentation ---")
    print(day_summary.to_string())
    
    # 5. By Season
    season_order = ["Winter", "Spring", "Summer", "Fall"]
    season_summary = df.groupby("Season").agg(
        Total_Shipments=("Shipment_ID", "count"),
        Delayed_Count=("Status", lambda x: (x == "Delayed").sum()),
        Primary_Delay_Rate=("Status", lambda x: round(100.0 * (x == "Delayed").sum() / x.isin(["Delivered", "Delayed"]).sum(), 2) if x.isin(["Delivered", "Delayed"]).sum() > 0 else 0.0),
        Operational_Failure_Rate=("Status", lambda x: round(100.0 * x.isin(["Delayed", "Lost", "Returned"]).sum() / len(x), 2)),
        Estimated_Concession_Exposure=("Estimated_Concession_Exposure", "sum")
    ).reindex(season_order)
    
    print("\n--- 5. Seasonal Performance Segmentation ---")
    print(season_summary.to_string())
    
    # 6. Top 10 High-Delay Routes (minimum 10 shipments)
    route_summary = resolved.groupby("Route").agg(
        Route_Volume=("Shipment_ID", "count"),
        Avg_Distance=("Distance_miles", "mean"),
        Delayed_Count=("Status", lambda x: (x == "Delayed").sum()),
        Primary_Delay_Rate=("Status", lambda x: round(100.0 * (x == "Delayed").mean(), 2))
    ).query("Route_Volume >= 10").sort_values("Primary_Delay_Rate", ascending=False).head(10)
    
    print("\n--- 6. Top 10 High-Delay Routes (Resolved Volume >= 10) ---")
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
    Identifies multi-variable interaction bottlenecks (Carrier x Warehouse, Carrier x Distance Tier).
    """
    print("\n" + "=" * 65)
    print("STEP 4: HIGH-DELAY COMBINATIONS & OPERATIONAL BOTTLENECK DISCOVERY")
    print("=" * 65)
    
    resolved = df[df["Status"].isin(["Delivered", "Delayed"])].copy()
    
    pivot_comb = resolved.groupby(["Origin_Warehouse", "Carrier"]).agg(
        Resolved_Volume=("Shipment_ID", "count"),
        Delayed_Count=("Status", lambda x: (x == "Delayed").sum()),
        Primary_Delay_Rate=("Status", lambda x: round(100.0 * (x == "Delayed").mean(), 1))
    ).reset_index()
    
    high_delay_comb = pivot_comb[
        (pivot_comb["Resolved_Volume"] >= 8) & (pivot_comb["Primary_Delay_Rate"] >= 15.0)
    ].sort_values("Primary_Delay_Rate", ascending=False)
    
    print("\n--- Critical Carrier x Warehouse Hotspots (Primary Delay Rate >= 15%) ---")
    print(high_delay_comb.to_string(index=False))
    
    long_haul = resolved[resolved["Distance_Tier"] == "Long-Haul (>1500 mi)"].groupby("Carrier").agg(
        Resolved_Volume=("Shipment_ID", "count"),
        Delayed_Count=("Status", lambda x: (x == "Delayed").sum()),
        Primary_Delay_Rate=("Status", lambda x: round(100.0 * (x == "Delayed").mean(), 1))
    ).sort_values("Primary_Delay_Rate", ascending=False)
    
    print("\n--- Long-Haul (>1500 mi) Carrier Reliability ---")
    print(long_haul.to_string())
    
    return high_delay_comb


def detect_unusual_spikes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies temporal anomaly weeks where delays spiked significantly above average.
    """
    print("\n--- Temporal Anomaly & Spike Detection ---")
    df_temp = df.copy()
    df_temp["Shipment_Week"] = df_temp["Shipment_Date"].dt.to_period("W").dt.start_time
    
    weekly = df_temp.groupby("Shipment_Week").agg(
        Weekly_Volume=("Shipment_ID", "count"),
        Delayed_Count=("Status", lambda x: (x == "Delayed").sum()),
        Operational_Failures=("Status", lambda x: x.isin(["Delayed", "Lost", "Returned"]).sum())
    ).reset_index()
    
    weekly["Delay_ZScore"] = (weekly["Delayed_Count"] - weekly["Delayed_Count"].mean()) / weekly["Delayed_Count"].std()
    spikes = weekly[weekly["Delay_ZScore"] > 1.8].sort_values("Delay_ZScore", ascending=False)
    
    print(f"Detected {len(spikes)} significant weekly operational shock spikes:")
    for _, row in spikes.iterrows():
        week_str = row["Shipment_Week"].strftime("%Y-%m-%d")
        print(f"   Week of {week_str}: {int(row['Delayed_Count'])} delays ({row['Weekly_Volume']} total volume), Z-Score: {row['Delay_ZScore']:.2f}")
        
    return spikes


def conduct_statistical_tests(df: pd.DataFrame):
    """
    Runs hypothesis tests to check for statistical association between operational factors and delay status.
    Uses pure numpy calculations to ensure environment independence without C-extension DLL policy blocks.
    """
    print("\n" + "=" * 65)
    print("STATISTICAL HYPOTHESIS TESTING (ASSOCIATION TESTS)")
    print("=" * 65)
    
    resolved = df[df["Status"].isin(["Delivered", "Delayed"])].copy()
    
    def calc_chi2(ctable):
        obs = ctable.values.astype(float)
        row_sums = obs.sum(axis=1, keepdims=True)
        col_sums = obs.sum(axis=0, keepdims=True)
        total = obs.sum()
        expected = (row_sums @ col_sums) / total
        chi2_stat = np.sum(((obs - expected) ** 2) / expected)
        dof = (obs.shape[0] - 1) * (obs.shape[1] - 1)
        return chi2_stat, dof

    # 1. Chi-Square Test: Carrier vs Delay Status
    contingency_carrier = pd.crosstab(resolved["Carrier"], resolved["Status"])
    chi2_car, dof_car = calc_chi2(contingency_carrier)
    print(f"1. Chi-Square Test of Association (Carrier vs Delay Status):")
    print(f"   Chi2 Statistic = {chi2_car:.3f} (df = {dof_car})")
    print(f"   Interpretation: {'Association Observed (Chi2 > df)' if chi2_car > dof_car else 'Low Association'}")
    
    # 2. Chi-Square Test: Warehouse vs Delay Status
    contingency_wh = pd.crosstab(resolved["Origin_Warehouse"], resolved["Status"])
    chi2_wh, dof_wh = calc_chi2(contingency_wh)
    print(f"\n2. Chi-Square Test of Association (Warehouse vs Delay Status):")
    print(f"   Chi2 Statistic = {chi2_wh:.3f} (df = {dof_wh})")
    print(f"   Interpretation: {'Association Observed (Chi2 > df)' if chi2_wh > dof_wh else 'Low Association'}")
    
    # 3. Kruskal-Wallis Test: Promised SLA Days across Distance Tiers (Secondary Analytical Outcome)
    sla_col = "Promised_SLA_Days" if "Promised_SLA_Days" in df.columns else "Distance_miles"
    groups = [group[sla_col].dropna().values for _, group in df.groupby("Distance_Tier")]
    all_vals = np.concatenate(groups)
    ranks = pd.Series(all_vals).rank().values
    
    N = len(all_vals)
    idx = 0
    kw_sum = 0.0
    for g in groups:
        n_g = len(g)
        r_g = ranks[idx : idx + n_g]
        kw_sum += (r_g.sum() ** 2) / n_g
        idx += n_g
        
    kw_stat = (12.0 / (N * (N + 1))) * kw_sum - 3.0 * (N + 1)
    print(f"\n3. Kruskal-Wallis Test (SLA Days across Distance Tiers - Secondary Extension):")
    print(f"   H-Statistic = {kw_stat:.3f} (df = {len(groups)-1})")
    print(f"   Interpretation: {'Significant Difference in Transit Distribution Across Tiers' if kw_stat > len(groups) else 'No Significant Difference'}")




def generate_and_save_visualizations(df: pd.DataFrame, output_dir: str = "reports/figures"):
    """
    Generates high-resolution visualization charts for reports and dashboard.
    """
    os.makedirs(output_dir, exist_ok=True)
    setup_visualization_style()
    print(f"\n-> Generating visualization figures in {output_dir}...")
    
    resolved = df[df["Status"].isin(["Delivered", "Delayed"])].copy()
    
    # 1. Carrier On-Time Delivery %
    carrier_agg = resolved.groupby("Carrier")["Status"].apply(
        lambda x: 100.0 * (x == "Delivered").mean()
    ).sort_values(ascending=True)
    
    plt.figure(figsize=(10, 5))
    bars = plt.barh(carrier_agg.index, carrier_agg.values, color="#2b5c8f", alpha=0.85)
    plt.axvline(90, color="#d9534f", linestyle="--", linewidth=1.5, label="90% Project Reference Target")
    plt.xlabel("On-Time Delivery Rate (%) [Delivered / Resolved]")
    plt.xlim(70, 100)
    plt.title("Carrier On-Time Delivery Reliability (%)")
    plt.legend()
    
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.3, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=9, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "otd_by_carrier.png"), dpi=300)
    plt.close()
    
    # 2. Warehouse Performance Bar Chart
    wh_agg = resolved.groupby("Origin_Warehouse")["Status"].apply(
        lambda x: 100.0 * (x == "Delivered").mean()
    ).sort_values(ascending=True)
    
    plt.figure(figsize=(10, 5))
    palette = ["#d9534f" if v < 85 else "#f0ad4e" if v < 90 else "#5cb85c" for v in wh_agg.values]
    bars = plt.barh(wh_agg.index, wh_agg.values, color=palette)
    plt.axvline(90, color="gray", linestyle="--", alpha=0.7)
    plt.xlabel("On-Time Delivery Rate (%) [Delivered / Resolved]")
    plt.title("Origin Warehouse Dispatch Reliability (%)")
    plt.xlim(75, 100)
    
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.3, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=9)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "otd_by_warehouse.png"), dpi=300)
    plt.close()
    
    # 3. Carrier x Warehouse Primary Delay Rate Heatmap
    pivot_heatmap = resolved.pivot_table(
        index="Origin_Warehouse", 
        columns="Carrier", 
        values="Status", 
        aggfunc=lambda x: round(100.0 * (x == "Delayed").mean(), 1)
    )
    plt.figure(figsize=(11, 7))
    sns.heatmap(pivot_heatmap, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={'label': 'Primary Delay Rate (%)'})
    plt.title("Operational Hotspot Matrix: Delay Rate (%) by Warehouse & Carrier")
    plt.xlabel("Carrier")
    plt.ylabel("Origin Warehouse")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "carrier_warehouse_heatmap.png"), dpi=300)
    plt.close()
    
    # 4. Day of Week & Seasonality Delay Rate Heatmap
    day_season = resolved.pivot_table(
        index="Shipment_DayOfWeek", 
        columns="Season", 
        values="Status", 
        aggfunc=lambda x: round(100.0 * (x == "Delayed").mean(), 1)
    ).reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])[["Winter", "Spring", "Summer", "Fall"]]
    
    plt.figure(figsize=(9, 5))
    sns.heatmap(day_season, annot=True, fmt=".1f", cmap="Blues", cbar_kws={'label': 'Primary Delay Rate (%)'})
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
    
    os.makedirs("reports", exist_ok=True)
    high_delay.to_csv("reports/operational_bottlenecks_summary.csv", index=False)
    print("\n-> Saved bottleneck summary to reports/operational_bottlenecks_summary.csv")


if __name__ == "__main__":
    run_eda()

