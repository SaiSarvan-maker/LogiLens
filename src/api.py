"""
LogiLens - Logistics Operations & Delay Prediction API
------------------------------------------------------
Lightweight FastAPI/Starlette microservice exposing operational KPIs,
bottleneck data, and the live Machine Learning delay risk estimator.
"""

import os
import json
import joblib
import pandas as pd
from typing import Optional
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route


# Global cached resources
_MODEL_PIPELINE = None
_ANALYTICS_DF = None


def get_model():
    """Lazy loader for serialized ML pipeline."""
    global _MODEL_PIPELINE
    if _MODEL_PIPELINE is None:
        model_path = os.path.join("models", "delay_risk_model.joblib")
        if os.path.exists(model_path):
            _MODEL_PIPELINE = joblib.load(model_path)
    return _MODEL_PIPELINE


def get_data() -> pd.DataFrame:
    """Lazy loader for processed analytics dataset."""
    global _ANALYTICS_DF
    if _ANALYTICS_DF is None:
        data_path = os.path.join("data", "processed", "delivery_operations_analytics.csv")
        if os.path.exists(data_path):
            _ANALYTICS_DF = pd.read_csv(data_path)
    return _ANALYTICS_DF


async def root(request):
    """API health check and metadata."""
    return JSONResponse({
        "service": "LogiLens Delivery Operations Analytics API",
        "version": "1.0.0",
        "status": "online",
        "description": "Enterprise operations API for delivery delay estimation and logistics KPI monitoring."
    })


async def get_kpis(request):
    """Returns top-level operations delay metrics and KPIs."""
    df = get_data()
    if df is None:
        return JSONResponse({"error": "Analytics dataset not generated yet."}, status_code=500)
    
    total = len(df)
    status_col = "Delivery_Status" if "Delivery_Status" in df.columns else "Status"
    resolved_df = df[df[status_col].isin(["Delivered", "Delayed"])]
    resolved_total = len(resolved_df)
    on_time = int((df[status_col] == "Delivered").sum())
    delayed = int((df[status_col] == "Delayed").sum())
    lost = int((df[status_col] == "Lost").sum())
    returned = int((df[status_col] == "Returned").sum())
    
    primary_delay_rate = round((delayed / resolved_total) * 100.0, 2) if resolved_total > 0 else 0.0
    op_failure_rate = round(((delayed + lost + returned) / total) * 100.0, 2)
    avg_delay = round(float(df[df[status_col] == "Delayed"]["Delay_Days"].mean()), 2) if delayed > 0 else 0.0
    concession_col = "Estimated_Concession_Exposure" if "Estimated_Concession_Exposure" in df.columns else "Delay_Penalty_Cost"
    exposure_cost = round(float(df[concession_col].sum()), 2)
    imputed_costs = int(df["Cost_Imputed"].sum()) if "Cost_Imputed" in df.columns else 0
    
    return JSONResponse({
        "total_shipments": total,
        "resolved_deliveries_denominator": resolved_total,
        "on_time_shipments": on_time,
        "delayed_shipments": delayed,
        "lost_shipments": lost,
        "returned_shipments": returned,
        "primary_status_delay_rate_pct": primary_delay_rate,
        "operational_failure_rate_pct": op_failure_rate,
        "average_delay_days_secondary_sla": avg_delay,
        "estimated_concession_exposure_usd": exposure_cost,
        "cost_imputed_records": imputed_costs,
        "note": "Primary metrics calculated on ground-truth Delivery_Status. Concession exposure is an assumed scenario estimate."
    })


async def get_bottlenecks(request):
    """Returns top high-delay routes and carrier reliability rankings."""
    df = get_data()
    if df is None:
        return JSONResponse({"error": "Dataset not available."}, status_code=500)
        
    status_col = "Delivery_Status" if "Delivery_Status" in df.columns else "Status"
    resolved = df[df[status_col].isin(["Delivered", "Delayed"])].copy()
    
    carrier_rankings = resolved.groupby("Carrier").agg(
        volume=("Shipment_ID", "count"),
        delayed_count=(status_col, lambda x: (x == "Delayed").sum()),
        primary_delay_rate=(status_col, lambda x: round(100.0 * (x == "Delayed").mean(), 2)),
        on_time_rate=(status_col, lambda x: round(100.0 * (x == "Delivered").mean(), 2))
    ).sort_values("primary_delay_rate", ascending=False).reset_index().to_dict(orient="records")
    
    top_routes = resolved.groupby("Route").agg(
        volume=("Shipment_ID", "count"),
        delayed_count=(status_col, lambda x: (x == "Delayed").sum()),
        primary_delay_rate=(status_col, lambda x: round(100.0 * (x == "Delayed").mean(), 2))
    ).query("volume >= 8").sort_values("primary_delay_rate", ascending=False).head(5).reset_index().to_dict(orient="records")
    
    return JSONResponse({
        "carrier_reliability": carrier_rankings,
        "top_bottleneck_routes": top_routes
    })


async def predict_delay(request):
    """Predicts delivery delay probability for an incoming shipment."""
    pipeline = get_model()
    if pipeline is None:
        return JSONResponse({"error": "Model artifact not found. Run src/05_delay_risk_model.py first."}, status_code=500)
    
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
    
    # Required feature extraction with safe defaults
    origin = body.get("Origin_Warehouse", "Warehouse_CHI")
    carrier = body.get("Carrier", "UPS")
    dest = body.get("Destination", "New York")
    prod = body.get("Product_Type", "Consumer Electronics")
    priority = body.get("Priority", "Standard")
    day = body.get("Shipment_DayOfWeek", "Monday")
    season = body.get("Season", "Winter")
    distance = float(body.get("Distance_miles", 1000.0))
    weight = float(body.get("Weight_kg", 20.0))
    cost = float(body.get("Cost", 150.0))
    sla_days = int(body.get("Promised_SLA_Days", 3))
    
    if distance < 500:
        dist_tier = "Short-Haul (<500 mi)"
    elif distance <= 1500:
        dist_tier = "Medium-Haul (500-1500 mi)"
    else:
        dist_tier = "Long-Haul (>1500 mi)"
        
    sample = {
        "Origin_Warehouse": origin,
        "Carrier": carrier,
        "Destination": dest,
        "Product_Type": prod,
        "Priority": priority,
        "Shipment_DayOfWeek": day,
        "Season": season,
        "Distance_Tier": dist_tier,
        "Distance_miles": distance,
        "Weight_kg": weight,
        "Cost": cost,
        "Promised_SLA_Days": sla_days
    }
    
    df_sample = pd.DataFrame([sample])
    prob_delay = float(pipeline.predict_proba(df_sample)[0, 1])
    is_delayed = bool(prob_delay >= 0.50)
    
    if prob_delay < 0.20:
        risk_level = "Low Risk"
        recommendation = "Standard dispatch. Route operates with high historical reliability."
    elif prob_delay < 0.45:
        risk_level = "Moderate Risk"
        recommendation = "Moderate friction risk. Recommend morning hub dispatch to avoid cutoff lag."
    else:
        risk_level = "High Risk"
        recommendation = "High risk of delivery delay. Suggest alternative carrier or expedited service tier."
        
    return JSONResponse({
        "status": "success",
        "input_shipment": sample,
        "delay_probability": round(prob_delay, 3),
        "delay_probability_percentage": f"{prob_delay*100:.1f}%",
        "predicted_outcome": "Delayed" if is_delayed else "On-Time",
        "risk_level": risk_level,
        "actionable_recommendation": recommendation
    })


routes = [
    Route("/", root, methods=["GET"]),
    Route("/kpis", get_kpis, methods=["GET"]),
    Route("/bottlenecks", get_bottlenecks, methods=["GET"]),
    Route("/predict-delay", predict_delay, methods=["POST"]),
]

app = Starlette(debug=True, routes=routes)

if __name__ == "__main__":
    import uvicorn
    print("Starting LogiLens API on http://127.0.0.1:8000 ...")
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=True)

