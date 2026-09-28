"""
LogiLens - Relational Enterprise Data Generator
------------------------------------------------
Generates normalized, relational data tables representing the operational
sources (Orders, Shipments, Warehouses, Carriers, Delivery Events)
aligned with the raw shipment extract.
"""

import pandas as pd
import numpy as np

def generate_relational_tables():
    print("Loading base dataset...")
    base_df = pd.read_csv("data/raw/logistics_shipments_dataset.csv")
    n_records = len(base_df)
    
    # Deterministic seed for reproducible human-like data
    np.random.seed(42)
    
    # ---------------------------------------------------------
    # 1. Warehouses Reference Table
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
    df_warehouses.to_csv("data/raw/warehouses.csv", index=False)
    print(f"Created warehouses.csv with {len(df_warehouses)} records")
    
    wh_map = {row["Warehouse_Name"]: row["Warehouse_ID"] for row in warehouse_metadata}
    
    # ---------------------------------------------------------
    # 2. Carriers Reference Table
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
    df_carriers.to_csv("data/raw/carriers.csv", index=False)
    print(f"Created carriers.csv with {len(df_carriers)} records")
    
    car_map = {row["Carrier_Name"]: row["Carrier_ID"] for row in carrier_metadata}
    
    # ---------------------------------------------------------
    # 3. Orders Table
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
        
        # Product type & Priority
        # Heuristic: perishables correlate with slightly lower weight, auto parts with higher weight
        if row["Weight_kg"] > 50:
            prod_choice = np.random.choice(["Industrial & Auto Parts", "Home & Kitchen", "Consumer Electronics"], p=[0.5, 0.3, 0.2])
        elif row["Weight_kg"] < 5:
            prod_choice = np.random.choice(["Healthcare & Medical", "Apparel & Footwear", "Consumer Electronics"], p=[0.4, 0.35, 0.25])
        else:
            prod_choice = np.random.choice(product_categories, p=product_probs)
            
        priority = np.random.choice(priorities, p=priority_probs)
        
        # Order Date is 0 to 2 days before Shipment Date
        shipment_dt = pd.to_datetime(row["Shipment_Date"])
        order_lead_days = np.random.choice([0, 1, 2], p=[0.2, 0.6, 0.2])
        order_dt = shipment_dt - pd.Timedelta(days=order_lead_days)
        order_date_str = order_dt.strftime("%Y-%m-%d")
        
        # Order Value based on product category & weight
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
        
        # Shipments row
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
        
        # Delivery Events
        # Target transit SLA: Transit_Days in the raw table represents the committed SLA
        target_sla_days = max(1, int(row["Transit_Days"]))
        promised_delivery_dt = shipment_dt + pd.Timedelta(days=target_sla_days)
        promised_delivery_str = promised_delivery_dt.strftime("%Y-%m-%d")
        
        # Exception reason
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
            "Promised_Delivery_Date": promised_delivery_str,
            "Actual_Delivery_Date": row["Delivery_Date"],
            "Delivery_Status": status,
            "Exception_Reason": exception
        })
        
    df_orders = pd.DataFrame(orders)
    df_shipments = pd.DataFrame(shipments)
    df_events = pd.DataFrame(delivery_events)
    
    df_orders.to_csv("data/raw/orders.csv", index=False)
    df_shipments.to_csv("data/raw/shipments.csv", index=False)
    df_events.to_csv("data/raw/delivery_events.csv", index=False)
    
    print(f"Created orders.csv ({len(df_orders)} rows)")
    print(f"Created shipments.csv ({len(df_shipments)} rows)")
    print(f"Created delivery_events.csv ({len(df_events)} rows)")
    print("All enterprise relational source tables generated successfully!")

if __name__ == "__main__":
    generate_relational_tables()
