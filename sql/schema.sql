-- ====================================================================
-- LogiLens: Logistics Operations & Delivery Analytics Schema (DDL)
-- ====================================================================

-- 1. Warehouses Dimension Table
CREATE TABLE IF NOT EXISTS dim_warehouses (
    warehouse_id VARCHAR(10) PRIMARY KEY,
    warehouse_name VARCHAR(50) NOT NULL,
    city VARCHAR(50) NOT NULL,
    state VARCHAR(10) NOT NULL,
    region VARCHAR(30) NOT NULL,
    capacity_tier VARCHAR(20),
    daily_capacity INT,
    dispatch_sla_hours INT
);

-- 2. Carriers Dimension Table
CREATE TABLE IF NOT EXISTS dim_carriers (
    carrier_id VARCHAR(10) PRIMARY KEY,
    carrier_name VARCHAR(50) NOT NULL,
    service_tier VARCHAR(50),
    fleet_type VARCHAR(50),
    base_rate_per_mile DECIMAL(6, 3),
    target_sla_days_per_1000mi DECIMAL(4, 2)
);

-- 3. Orders Dimension Table
CREATE TABLE IF NOT EXISTS fact_orders (
    order_id VARCHAR(20) PRIMARY KEY,
    customer_id VARCHAR(20) NOT NULL,
    order_date DATE NOT NULL,
    product_type VARCHAR(50) NOT NULL,
    priority VARCHAR(20) NOT NULL,
    order_quantity INT NOT NULL,
    order_value_usd DECIMAL(10, 2) NOT NULL
);

-- 4. Shipments Fact Table
CREATE TABLE IF NOT EXISTS fact_shipments (
    shipment_id VARCHAR(20) PRIMARY KEY,
    order_id VARCHAR(20) NOT NULL,
    warehouse_id VARCHAR(10) NOT NULL,
    carrier_id VARCHAR(10) NOT NULL,
    shipment_date DATE NOT NULL,
    weight_kg DECIMAL(8, 2),
    cost DECIMAL(10, 2),
    FOREIGN KEY (order_id) REFERENCES fact_orders(order_id),
    FOREIGN KEY (warehouse_id) REFERENCES dim_warehouses(warehouse_id),
    FOREIGN KEY (carrier_id) REFERENCES dim_carriers(carrier_id)
);

-- 5. Delivery Events Tracking Table
CREATE TABLE IF NOT EXISTS fact_delivery_events (
    event_id VARCHAR(20) PRIMARY KEY,
    shipment_id VARCHAR(20) NOT NULL,
    destination_city VARCHAR(50) NOT NULL,
    distance_miles INT NOT NULL,
    promised_sla_days INT NOT NULL,
    promised_delivery_date DATE NOT NULL,
    actual_delivery_date DATE,
    delivery_status VARCHAR(30) NOT NULL,
    exception_reason VARCHAR(100),
    FOREIGN KEY (shipment_id) REFERENCES fact_shipments(shipment_id)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_shipment_date ON fact_shipments(shipment_date);
CREATE INDEX IF NOT EXISTS idx_delivery_status ON fact_delivery_events(delivery_status);
CREATE INDEX IF NOT EXISTS idx_shipment_carrier ON fact_shipments(carrier_id);
CREATE INDEX IF NOT EXISTS idx_shipment_warehouse ON fact_shipments(warehouse_id);
