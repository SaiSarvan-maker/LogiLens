"""
LogiLens - Logistics Operations & Delay Prediction API
------------------------------------------------------
Lightweight FastAPI microservice exposing operational KPIs,
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
    delivered = len(df[df["Status"].isin(["Delivered", "Delayed"])])
    on_time = int(df["Is_On_Time"].sum())
    delayed = int(df["Is_Delayed"].sum())
    otd_pct = round((on_time / delivered) * 100.0, 2) if delivered > 0 else 0.0
    avg_delay = round(float(df[df["Is_Delayed"] == 1]["Delay_Days"].mean()), 2)
    delay_cost = round(float(df["Delay_Penalty_Cost"].sum()), 2)
    
    return JSONResponse({
        "total_shipments": total,
        "resolved_shipments": delivered,
        "on_time_shipments": on_time,
        "delayed_shipments": delayed,
        "on_time_delivery_rate_pct": otd_pct,
        "average_delay_days": avg_delay,
        "total_estimated_penalty_cost_usd": delay_cost
    })


async def get_bottlenecks(request):
    """Returns top high-delay routes and carrier reliability rankings."""
    df = get_data()
    if df is None:
        return JSONResponse({"error": "Dataset not available."}, status_code=500)
        
    carrier_rankings = df.groupby("Carrier").agg(
        volume=("Shipment_ID", "count"),
        otd_rate=("Is_On_Time", lambda x: round(100.0 * x.mean(), 2)),
        avg_delay=("Delay_Days", lambda x: round(x[x > 0].mean() if (x > 0).sum() > 0 else 0, 2))
    ).sort_values("otd_rate", ascending=True).reset_index().to_dict(orient="records")
    
    top_routes = df.groupby("Route").agg(
        volume=("Shipment_ID", "count"),
        delay_rate=("Is_Delayed", lambda x: round(100.0 * x.mean(), 2)),
        avg_delay_days=("Delay_Days", lambda x: round(x[x > 0].mean() if (x > 0).sum() > 0 else 0, 2))
    ).query("volume >= 10").sort_values("delay_rate", ascending=False).head(5).reset_index().to_dict(orient="records")
    
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
