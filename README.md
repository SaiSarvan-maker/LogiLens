# 🚚 LogiLens — Delivery Delay & Operations Analytics

> **End-to-End Logistics Analytics, Observed Delay Patterns, and Prototype Delay Risk Estimation**  
> LogiLens implements the Delivery Delay & Operations Analytics specification using a public shipment dataset and a documented synthetic relational extension where required source tables were unavailable.

---

## 📌 Executive Overview

In modern supply chain and e-commerce networks, late deliveries erode customer loyalty, depress Net Promoter Scores (NPS), and trigger customer concession exposure. **LogiLens** is an operations intelligence system designed to discover observed delay patterns, analyze operational failure factors, and equip dispatch managers with prototype delay-risk estimation.

Across **2,000 shipments, 10 regional fulfillment warehouses, and 7 national/regional carriers**, this project combines multi-source enterprise data, evaluates status-based delay metrics, pinpoints operational friction, and deploys an explainable Machine Learning model alongside an interactive Streamlit Control Tower.

---

## 🏗️ Data Architecture & Provenance

LogiLens maintains an honest data architecture separating observed ground-truth data from derived/synthetic extensions:

```
REAL SOURCE DATA (data/raw/)
      │
      └── logistics_shipments_dataset.csv       ← REAL SOURCE EXTRACT
      │
      ▼
SYNTHETIC EXTENSION (data/synthetic/)
      ├── orders.csv                            ← GENERATED RELATIONAL
      ├── shipments.csv                         ← GENERATED RELATIONAL
      ├── warehouses.csv                        ← GENERATED RELATIONAL
      ├── carriers.csv                          ← GENERATED RELATIONAL
      └── delivery_events.csv                   ← GENERATED RELATIONAL
      │
      ▼
PROCESSED ANALYTICS (data/processed/)
      ├── delivery_operations_analytics.csv     ← UNIFIED ANALYTICAL FACT
      ├── delivery_operations_analytics.parquet
      └── logistics.db                          ← INDEXED SQLITE DATABASE
```

### Analysis Options & Target Definitions:
- **Option B (Official Primary Analysis)**: Ground-truth status target (`Delivery_Status == "Delayed"`). Calculated on resolved shipments (`Delivered` + `Delayed`):
  $$\text{Primary Delay Rate} = \frac{\text{Delayed}}{\text{Delivered} + \text{Delayed}} \times 100$$
- **Operational Failure Rate**: Overall operational disruption metric including lost and returned packages:
  $$\text{Operational Failure Rate} = \frac{\text{Delayed} + \text{Lost} + \text{Returned}}{\text{Total Shipments}} \times 100$$
- **Option A (Secondary Analytical Extension)**: Analytical SLA expected baseline date derived from the `Transit_Days` field (`Derived_Promised_Delivery_Date`). Not treated as an observed contractual commitment.

---

## 📂 Repository Layout

```text
LogiLens/
├── dashboard/
│   └── app.py                      # Interactive Streamlit operations control tower
├── data/
│   ├── raw/                        # Real source shipment extract
│   │   └── logistics_shipments_dataset.csv
│   ├── synthetic/                  # Documented synthetic relational tables
│   │   ├── orders.csv
│   │   ├── shipments.csv
│   │   ├── warehouses.csv
│   │   ├── carriers.csv
│   │   └── delivery_events.csv
│   ├── processed/                  # Validated, enriched analytical datasets
│   │   ├── delivery_operations_analytics.csv
│   │   └── delivery_operations_analytics.parquet
│   └── logistics.db                # Indexed SQLite operational database
├── models/
│   ├── delay_risk_model.joblib     # Serialized Decision Tree ML pipeline
│   ├── model_metrics.json          # Precision, Recall, ROC-AUC, 5-Fold CV metrics
│   └── feature_names.json          # Preprocessed feature registry
├── notebooks/
│   └── 01_delivery_delay_analysis.ipynb # End-to-end demonstration notebook
├── reports/
│   ├── delivery_delay_executive_report.md # Formal supply chain leadership report
│   ├── operational_bottlenecks_summary.csv # High-delay combination matrix
│   └── figures/                    # Publication-quality charts (PNG)
│       ├── otd_by_carrier.png
│       ├── otd_by_warehouse.png
│       ├── carrier_warehouse_heatmap.png
│       ├── delay_by_day_and_season.png
│       ├── feature_importances.png
│       └── confusion_matrix.png
├── sql/
│   ├── schema.sql                  # Relational DDL with primary & foreign keys
│   ├── combine_logistics_data.sql  # Enterprise SQL join & status calculation
│   └── delay_analytics_queries.sql # Practical SQL operational queries
├── src/
│   ├── 01_data_audit.py            # Data audit & provenance inspection
│   ├── 02_data_validation.py       # Data quality & anomaly detection checks
│   ├── 03_data_pipeline.py         # Full ETL pipeline (joins, cleaning, metrics)
│   ├── 04_exploratory_analysis.py  # 6-dimension segmentation & statistical tests
│   ├── 05_delay_risk_model.py      # Leakage-free Decision Tree ML training
│   ├── api.py                      # FastAPI/Starlette microservice
│   └── generate_relational_data.py # Generator for synthetic relational tables
├── requirements.txt                # Python environment dependencies
└── README.md                       # Comprehensive documentation
```

---

## 🎯 Step-by-Step Implementation Guide

### Step 1: Ingestion & Honest Data Hygiene
- Combined real raw extract and synthetic relational sources while preserving data lineage.
- Data Quality Hygiene:
  - Preserved missing delivery dates without circular date creation via `Delivery_Date_Missing` indicator flag.
  - Imputed missing shipping costs via carrier mileage and weight formulas, explicitly flagged with `Cost_Imputed = 1`.
  - Corrected date inversion anomalies and invalid weight boundaries.

### Step 2: Compute Operations Metrics & Delay KPIs
- **Primary Status Target**: `Status == "Delayed"` (1 for Delayed, 0 for Delivered).
- **Primary Delay Rate**: $10.88\%$ across resolved deliveries ($201 / 1,847$).
- **Operational Failure Rate**: $11.25\%$ incorporating delayed, lost, and returned shipments.
- **Estimated Concession Exposure**: Scenario estimate based on an assumed concession formula (20% freight reimbursement + $15 administrative fee = $\$11,030.07$ total).

### Step 3: Segment Delays across 6 Core Dimensions
Executed in [`src/04_exploratory_analysis.py`](file:///c:/projects/Int-project/LogiLens/src/04_exploratory_analysis.py):
1. **Warehouse**: `Warehouse_NYC` recorded the lowest delay rate (**5.88%**), while `Warehouse_MIA` (**14.11%**) and `Warehouse_SF` (**13.56%**) experienced dispatch drag.
2. **Carrier**: `USPS` delivered lowest delay rate (**7.19%**), whereas `DHL` (**16.19%**) and `Amazon Logistics` (**14.17%**) had higher delay rates on long-haul corridors.
3. **Route**: Long-haul corridors suffered highest delay rates (e.g., `Los Angeles -> Houston`: **26.7% late**, `Miami -> Boston`: **25.0% late**).
4. **Product Type**: Handling-sensitive categories like `Perishables & Grocery` (**13.91%**) experienced higher delay frequencies.
5. **Day of Week**: Tuesday dispatches exhibited lowest reliability, reflecting weekend backlog accumulation.
6. **Season**: Winter dispatches recorded elevated delay rates.

### Step 4: Exploratory Bottleneck Discovery & Statistical Spikes
- **Critical Hotspot Combinations**:
  - `Warehouse_ATL + Amazon Logistics`: **21.2% delay rate** across 33 resolved shipments.
  - `Warehouse_HOU + Amazon Logistics`: **20.7% delay rate** across 29 resolved shipments.
- **Temporal Anomaly Tracking**: Using weekly Z-score analysis, detected 4 distinct operational shock weeks ($Z > 1.8$).
- **Statistical Testing**:
  - Chi-Square test of carrier/warehouse association with delay status.
  - Kruskal-Wallis test across transit day distributions.

### Step 5: Leakage-Free Machine Learning Delay Risk Estimator
Implemented in [`src/05_delay_risk_model.py`](file:///c:/projects/Int-project/LogiLens/src/05_delay_risk_model.py):
- **Leakage Prevention**: Strictly excluded outcome variables (`Delivery_Date`, `Days_Variance`, `Delay_Days`, `Estimated_Concession_Exposure`). Only features available prior to delivery were included (`Warehouse`, `Carrier`, `Destination`, `Product`, `Priority`, `Distance`, `Weight`, `Cost`, `SLA_Days`, `DayOfWeek`, `Season`).
- **Algorithm**: Interpretable Decision Tree Classifier (depth = 4) alongside Logistic Regression and Random Forest benchmarks.
- **Honest Evaluation**: Pre-dispatch attributes provide moderate predictive signal (ROC-AUC ≈ 0.53 - 0.57), demonstrating that unexpected operational disruptions account for significant remaining variance.

### Step 6: Streamlit Operations Dashboard & REST API
- Built [`dashboard/app.py`](file:///c:/projects/Int-project/LogiLens/dashboard/app.py) with two clear sections:
  - **Section A: Core Observed Metrics** (Direct Source Ground Truth).
  - **Section B: Analytical Extension** (Derived SLA, Delay Days, Estimated Concession Exposure with explicit note).
- Built [`src/api.py`](file:///c:/projects/Int-project/LogiLens/src/api.py) microservice (`GET /kpis`, `GET /bottlenecks`, `POST /predict-delay`).

---

## 🚀 How to Run the Project Locally

### 1. Prerequisites
Ensure you have Python 3.10+ installed:
```powershell
pip install -r requirements.txt
```

### 2. Run the Data Pipeline (ETL)
```powershell
python src/03_data_pipeline.py
```

### 3. Run Exploratory Bottleneck Analysis
```powershell
python src/04_exploratory_analysis.py
```

### 4. Train & Evaluate the Delay Risk ML Model
```powershell
python src/05_delay_risk_model.py
```

### 5. Launch the Interactive Streamlit Dashboard
```powershell
streamlit run dashboard/app.py
```
Open your browser at `http://localhost:8501`.

### 6. (Optional) Run the Operations REST API
```powershell
python src/api.py
```
Test endpoints at `http://localhost:8000/kpis` or `http://localhost:8000/predict-delay`.

---

## 💡 Viva & Presentation Talking Points

When presenting this project, frame your technical decisions with honest data science hygiene:

1. **Data Provenance**:
   *"I clearly separated the original shipment extract (`data/raw`) from the synthetic relational extension (`data/synthetic`). Rather than pretending all tables were observed ERP data, I explicitly documented the synthetic extension used for SQL demonstration."*
2. **Defensible Delay Target**:
   *"I used ground-truth `Delivery_Status == 'Delayed'` as my primary target and calculated delay rate strictly over resolved deliveries (`Delivered` + `Delayed`), while creating an Operational Failure Rate for lost and returned packages."*
3. **Traceable Imputation & Derived SLA**:
   *"I preserved missingness flags (`Delivery_Date_Missing`, `Cost_Imputed = 1`) so downstream users can distinguish observed values from estimated values, and framed promised delivery dates as a derived analytical baseline."*
4. **Data Leakage Prevention**:
   *"In the machine learning model, I strictly excluded post-delivery outcome features like delivery date and variance days. Evaluating on pre-dispatch features alone provided an honest model assessment."*

---

*LogiLens — Supply Chain & Logistics Analytics Intelligence.*
