"""
LogiLens - Relational Enterprise Data Generator (Synthetic Extension)
----------------------------------------------------------------------
Generates normalized, synthetic relational data tables (Orders, Shipments,
Warehouses, Carriers, Delivery Events) aligned with the raw shipment extract.

DATA PROVENANCE & ARCHITECTURE:
- REAL SOURCE DATA: data/raw/logistics_shipments_dataset.csv
- SYNTHETIC EXTENSION: data/synthetic/*.csv (Orders, Shipments, Warehouses, Carriers, Delivery Events)
- SLA DATES: Analytical expected SLA derived from Transit_Days duration field, not observed contractual dates.
"""

import os
import pandas as pd
import numpy as np

def generate_relational_tables(raw_dir: str = "data/raw", synthetic_dir: str = "data/synthetic"):
    os.makedirs(synthetic_dir, exist_ok=True)
    
    raw_path = os.path.join(raw_dir, "logistics_shipments_dataset.csv")
    print(f"Loading base dataset from {raw_path}...")
    base_df = pd.read_csv(raw_path)
    
    # Deterministic seed for reproducible data generation
    np.random.seed(42)
    
    # ---------------------------------------------------------
    # 1. Warehouses Reference Table (Synthetic)
    # ---------------------------------------------------------
    warehouse_metadata = [
        {"Warehouse_ID": "WH-ATL", "Warehouse_Name": "Warehouse_ATL", "City": "Atlanta", "State": "GA", "Region": "Southeast", "Capacity_Tier": "Tier-1", "Daily_Capacity": 12000, "Dispatch_SLA_Hours": 12},
        {"Warehouse_ID": "WH-BOS", "Warehouse_Name": "Warehouse_BOS", "City": "Boston", "State": "MA", "Region": "Northeast", "Capacity_Tier": "Tier-2", "Daily_Capacity": 8000, "Dispatch_SLA_Hours": 18},
        {"Warehouse_ID": "WH-CHI", "Warehouse_Name": "Warehouse_CHI", "City": "Chicago", "State": "IL", "Region": "Midwest", "Capacity_Tier": "Tier-1", "Daily_Capacity": 15000, "Dispatch_SLA_Hours": 12},
        {"Warehouse_ID": "WH-DEN", "Warehouse_Name": "Warehouse_DEN", "City": "Denver", "State": "CO", "Region": "Mountain", "Capacity_Tier": "Tier-2", "Daily_Capacity": 7500, "Dispatch_SLA_Hours": 18},
        {"Warehouse_ID": "WH-HOU", "Warehouse_Name": "Warehouse_HOU", "City": "Houston", "State": "TX", "Region": "South", "Capacity_Tier": "Tier-1", "Daily_Capacity": 11000, "Dispatch_SLA_Hours": 14},
        {"Warehouse_ID": "WH-LA",  "Warehouse_Name": "Warehouse_LA",  "City": "Los Angeles", "State": "CA", "Region": "West", "Capacity_Tier": "Tier-1", "Daily_Capacity": 18000, "Dispatch_SLA_Hours": 12},
        {"Warehouse_ID": "WH-MIA", "Warehouse_Name": "Warehouse_MIA", "City": "Miami", "State": "FL", "Region": "Southeast", "Capacity_Tier": "Tier-2", "Daily_Capacity": 8500, "Dispatch_SLA_Hours": 16},
        {"Warehouse_ID": "WH-NYC", "Warehouse_Name": "Warehouse_NYC", "City": "New York", "State": "NY", "Region": "Northeast", "Capacity_Tier": "Tier-1", "Daily_Capacity": 14000, "Dispatch_SLA_Hours": 14},
        {"Warehouse_ID": "WH-SEA", "Warehouse_Name": "Warehouse_SEA", "City": "Seattle", "State": "WA", "Region": "Northwest", "Capacity_Tier": "Tier-2", "Daily_Capacity": 9000, "Dispatch_SLA_Hours": 16},
        {"Warehouse_ID": "WH-SF",  "Warehouse_Name": "Warehouse_SF",  "City": "San Francisco", "State": "CA", "Region": "West", "Capacity_Tier": "Tier-2", "Daily_Capacity": 9500, "Dispatch_SLA_Hours": 16}
    ]
    df_warehouses = pd.DataFrame(warehouse_metadata)
    wh_path = os.path.join(synthetic_dir, "warehouses.csv")
    df_warehouses.to_csv(wh_path, index=False)
    print(f"Created {wh_path} with {len(df_warehouses)} records")
    
    wh_map = {row["Warehouse_Name"]: row["Warehouse_ID"] for row in warehouse_metadata}
    
    # ---------------------------------------------------------
    # 2. Carriers Reference Table (Synthetic)
    # ---------------------------------------------------------
    carrier_metadata = [
        {"Carrier_ID": "CAR-AMZ", "Carrier_Name": "Amazon Logistics", "Service_Tier": "Express/Standard", "Fleet_Type": "Dedicated Fleet", "Base_Rate_Per_Mile": 0.14, "Target_SLA_Days_Per_1000mi": 2.2},
        {"Carrier_ID": "CAR-DHL", "Carrier_Name": "DHL",              "Service_Tier": "Express",          "Fleet_Type": "Air & Ground Hub",  "Base_Rate_Per_Mile": 0.19, "Target_SLA_Days_Per_1000mi": 1.8},
        {"Carrier_ID": "CAR-FDX", "Carrier_Name": "FedEx",            "Service_Tier": "Priority Ground",  "Fleet_Type": "Hub & Spoke Network","Base_Rate_Per_Mile": 0.17, "Target_SLA_Days_Per_1000mi": 2.0},
        {"Carrier_ID": "CAR-LSR", "Carrier_Name": "LaserShip",        "Service_Tier": "Regional Ground",  "Fleet_Type": "Contracted Regional", "Base_Rate_Per_Mile": 0.12, "Target_SLA_Days_Per_1000mi": 2.8},
        {"Carrier_ID": "CAR-ONT", "Carrier_Name": "OnTrac",           "Service_Tier": "Regional Ground",  "Fleet_Type": "Regional Courier",   "Base_Rate_Per_Mile": 0.13, "Target_SLA_Days_Per_1000mi": 2.6},
        {"Carrier_ID": "CAR-UPS", "Carrier_Name": "UPS",              "Service_Tier": "Standard Ground",  "Fleet_Type": "National Integrated","Base_Rate_Per_Mile": 0.16, "Target_SLA_Days_Per_1000mi": 2.1},
        {"Carrier_ID": "CAR-USPS", "Carrier_Name": "USPS",             "Service_Tier": "Postal Ground",    "Fleet_Type": "Public Postal",      "Base_Rate_Per_Mile": 0.10, "Target_SLA_Days_Per_1000mi": 3.0}
    ]
    df_carriers = pd.DataFrame(carrier_metadata)
    car_path = os.path.join(synthetic_dir, "carriers.csv")
    df_carriers.to_csv(car_path, index=False)
    print(f"Created {car_path} with {len(df_carriers)} records")
    
    car_map = {row["Carrier_Name"]: row["Carrier_ID"] for row in carrier_metadata}
    
    # ---------------------------------------------------------
    # 3. Orders, Shipments, & Delivery Events Tables (Synthetic)
    # ---------------------------------------------------------
    product_categories = [
        "Consumer Electronics",
        "Apparel & Footwear",
        "Healthcare & Medical",
        "Home & Kitchen",
        "Perishables & Grocery",
        "Industrial & Auto Parts"
    ]
    product_probs = [0.25, 0.22, 0.14, 0.18, 0.11, 0.10]
    priorities = ["Standard", "Express", "Same-Day"]
    priority_probs = [0.65, 0.28, 0.07]
    
    orders = []
    shipments = []
    delivery_events = []
    
    for idx, row in base_df.iterrows():
        shipment_id = row["Shipment_ID"]
        order_id = f"ORD-{shipment_id.replace('SH', '')}"
        customer_id = f"CUST-{np.random.randint(10001, 15000)}"
        
        if row["Weight_kg"] > 50:
            prod_choice = np.random.choice(["Industrial & Auto Parts", "Home & Kitchen", "Consumer Electronics"], p=[0.5, 0.3, 0.2])
        elif row["Weight_kg"] < 5:
            prod_choice = np.random.choice(["Healthcare & Medical", "Apparel & Footwear", "Consumer Electronics"], p=[0.4, 0.35, 0.25])
        else:
            prod_choice = np.random.choice(product_categories, p=product_probs)
            
        priority = np.random.choice(priorities, p=priority_probs)
        
        shipment_dt = pd.to_datetime(row["Shipment_Date"])
        order_lead_days = np.random.choice([0, 1, 2], p=[0.2, 0.6, 0.2])
        order_dt = shipment_dt - pd.Timedelta(days=order_lead_days)
        order_date_str = order_dt.strftime("%Y-%m-%d")
        
        base_unit_price = {
            "Consumer Electronics": 220.0,
            "Apparel & Footwear": 65.0,
            "Healthcare & Medical": 140.0,
            "Home & Kitchen": 85.0,
            "Perishables & Grocery": 45.0,
            "Industrial & Auto Parts": 175.0
        }
        item_qty = max(1, int(np.random.poisson(2)))
        order_val = round(base_unit_price[prod_choice] * item_qty + np.random.uniform(5, 50), 2)
        
        orders.append({
            "Order_ID": order_id,
            "Customer_ID": customer_id,
            "Order_Date": order_date_str,
            "Product_Type": prod_choice,
            "Priority": priority,
            "Order_Quantity": item_qty,
            "Order_Value_USD": order_val
        })
        
        wh_id = wh_map.get(row["Origin_Warehouse"], "WH-CHI")
        car_id = car_map.get(row["Carrier"], "CAR-UPS")
        
        shipments.append({
            "Shipment_ID": shipment_id,
            "Order_ID": order_id,
            "Warehouse_ID": wh_id,
            "Carrier_ID": car_id,
            "Shipment_Date": row["Shipment_Date"],
            "Weight_kg": row["Weight_kg"],
            "Cost": row["Cost"]
        })
        
        # Target transit SLA: Transit_Days in the raw table represents the committed baseline SLA
        target_sla_days = max(1, int(row["Transit_Days"]))
        derived_promised_dt = shipment_dt + pd.Timedelta(days=target_sla_days)
        derived_promised_str = derived_promised_dt.strftime("%Y-%m-%d")
        
        status = row["Status"]
        if status == "Delayed":
            exception = np.random.choice(
                ["Weather Disruption", "Carrier Hub Congestion", "Last-Mile Mechanical Failure", "Routing Deviation", "High Holiday Volume"],
                p=[0.30, 0.35, 0.15, 0.10, 0.10]
            )
        elif status == "Lost":
            exception = "Lost in Hub Transit"
        elif status == "Returned":
            exception = "Customer Refused / Delivery Failed"
        elif status == "In Transit":
            exception = "Active Network Transit"
        else:
            exception = "Normal Delivery"
            
        event_id = f"EVT-{100000 + idx}"
        delivery_events.append({
            "Event_ID": event_id,
            "Shipment_ID": shipment_id,
            "Destination_City": row["Destination"],
            "Distance_miles": row["Distance_miles"],
            "Promised_SLA_Days": target_sla_days,
            "Derived_Promised_Delivery_Date": derived_promised_str,
            "Actual_Delivery_Date": row["Delivery_Date"],
            "Delivery_Status": status,
            "Exception_Reason": exception
        })
        
    df_orders = pd.DataFrame(orders)
    df_shipments = pd.DataFrame(shipments)
    df_events = pd.DataFrame(delivery_events)
    
    df_orders.to_csv(os.path.join(synthetic_dir, "orders.csv"), index=False)
    df_shipments.to_csv(os.path.join(synthetic_dir, "shipments.csv"), index=False)
    df_events.to_csv(os.path.join(synthetic_dir, "delivery_events.csv"), index=False)
    
    print(f"Created synthetic/orders.csv ({len(df_orders)} rows)")
    print(f"Created synthetic/shipments.csv ({len(df_shipments)} rows)")
    print(f"Created synthetic/delivery_events.csv ({len(df_events)} rows)")
    print("All synthetic relational extension tables generated successfully in data/synthetic/")

if __name__ == "__main__":
    generate_relational_tables()

