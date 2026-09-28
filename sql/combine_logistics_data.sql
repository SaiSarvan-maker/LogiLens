-- ====================================================================
-- LogiLens: Combine Order, Shipment, Warehouse, Carrier, and Delivery Events
-- Step 1 & 2 of Analytics Specification
-- ====================================================================

WITH base_joined AS (
    SELECT
        s.shipment_id,
        o.order_id,
        o.customer_id,
        o.order_date,
        o.product_type,
        o.priority,
        o.order_quantity,
        o.order_value_usd,
        
        -- Warehouse details
        w.warehouse_id,
        w.warehouse_name,
        w.city AS origin_city,
        w.state AS origin_state,
        w.region AS origin_region,
        w.capacity_tier,
        
        -- Carrier details
        c.carrier_id,
        c.carrier_name,
        c.service_tier,
        c.base_rate_per_mile,
        
        -- Shipment financials & physicals
        s.shipment_date,
        s.weight_kg,
        s.cost,
        
        -- Delivery event details
        e.destination_city,
        e.distance_miles,
        e.promised_sla_days,
        e.promised_delivery_date,
        e.actual_delivery_date,
        e.delivery_status,
        e.exception_reason
    FROM fact_shipments s
    INNER JOIN fact_orders o ON s.order_id = o.order_id
    INNER JOIN dim_warehouses w ON s.warehouse_id = w.warehouse_id
    INNER JOIN dim_carriers c ON s.carrier_id = c.carrier_id
    INNER JOIN fact_delivery_events e ON s.shipment_id = e.shipment_id
)
SELECT
    shipment_id,
    order_id,
    customer_id,
    warehouse_name AS origin_warehouse,
    origin_city,
    origin_state,
    origin_region,
    destination_city,
    (origin_city || ' -> ' || destination_city) AS route,
    carrier_name AS carrier,
    service_tier,
    product_type,
    priority,
    order_quantity,
    order_value_usd,
    weight_kg,
    cost,
    distance_miles,
    CASE 
        WHEN distance_miles < 500 THEN 'Short-Haul (<500 mi)'
        WHEN distance_miles <= 1500 THEN 'Medium-Haul (500-1500 mi)'
        ELSE 'Long-Haul (>1500 mi)'
    END AS distance_tier,
    shipment_date,
    promised_delivery_date,
    actual_delivery_date,
    delivery_status,
    exception_reason,
    
    -- Delay calculation:
    -- Delay days is positive if actual delivery date is after promised delivery date
    CASE 
        WHEN actual_delivery_date IS NULL THEN NULL
        WHEN julianday(actual_delivery_date) > julianday(promised_delivery_date) 
            THEN CAST(ROUND(julianday(actual_delivery_date) - julianday(promised_delivery_date)) AS INT)
        ELSE 0
    END AS delay_days,
    
    -- On-time indicator (1 = On-Time, 0 = Delayed/Failed)
    CASE 
        WHEN delivery_status = 'Delivered' 
             AND (actual_delivery_date <= promised_delivery_date OR actual_delivery_date IS NULL) THEN 1
        WHEN delivery_status = 'Delayed' THEN 0
        WHEN actual_delivery_date > promised_delivery_date THEN 0
        ELSE 0
    END AS is_on_time,
    
    -- Delay indicator (1 = Delayed, 0 = On-Time)
    CASE 
        WHEN delivery_status = 'Delayed' OR actual_delivery_date > promised_delivery_date THEN 1
        ELSE 0
    END AS is_delayed,
    
    -- Cost per mile
    ROUND(cost / CAST(distance_miles AS FLOAT), 3) AS cost_per_mile
FROM base_joined;
