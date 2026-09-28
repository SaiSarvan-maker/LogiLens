-- ====================================================================
-- LogiLens: Operational Delay Segmentation Queries
-- Step 3 of Analytics Specification
-- ====================================================================

-- 1. On-Time Delivery % (OTD%) and Delay Metrics by Carrier
SELECT
    carrier,
    COUNT(*) AS total_shipments,
    SUM(CASE WHEN is_on_time = 1 THEN 1 ELSE 0 END) AS on_time_shipments,
    SUM(is_delayed) AS delayed_shipments,
    ROUND(100.0 * SUM(CASE WHEN is_on_time = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS on_time_percentage,
    ROUND(AVG(delay_days), 2) AS avg_delay_days,
    ROUND(AVG(cost_per_mile), 3) AS avg_cost_per_mile
FROM delivery_operations_analytics
WHERE actual_delivery_date IS NOT NULL
GROUP BY carrier
ORDER BY on_time_percentage ASC;

-- 2. Performance and Delay Hotspots by Origin Warehouse
SELECT
    origin_warehouse,
    origin_region,
    COUNT(*) AS total_shipments,
    ROUND(100.0 * SUM(is_on_time) / COUNT(*), 2) AS on_time_percentage,
    ROUND(AVG(delay_days), 2) AS avg_delay_days,
    SUM(CASE WHEN is_delayed = 1 THEN 1 ELSE 0 END) AS delayed_orders
FROM delivery_operations_analytics
GROUP BY origin_warehouse, origin_region
ORDER BY on_time_percentage ASC;

-- 3. Top 10 High-Delay Routes (Corridors)
SELECT
    route,
    COUNT(*) AS route_volume,
    ROUND(AVG(distance_miles), 0) AS avg_distance,
    SUM(is_delayed) AS delayed_count,
    ROUND(100.0 * SUM(is_delayed) / COUNT(*), 2) AS delay_rate_pct,
    ROUND(AVG(delay_days), 2) AS avg_delay_days
FROM delivery_operations_analytics
GROUP BY route
HAVING COUNT(*) >= 10
ORDER BY delay_rate_pct DESC, avg_delay_days DESC
LIMIT 10;

-- 4. Delay Segmentation by Product Type
SELECT
    product_type,
    COUNT(*) AS total_volume,
    ROUND(AVG(weight_kg), 1) AS avg_weight,
    ROUND(100.0 * SUM(is_on_time) / COUNT(*), 2) AS on_time_percentage,
    ROUND(AVG(delay_days), 2) AS avg_delay_days
FROM delivery_operations_analytics
GROUP BY product_type
ORDER BY on_time_percentage ASC;

-- 5. Day of Week Dispatch Friction
SELECT
    shipment_dayofweek,
    COUNT(*) AS shipments_dispatched,
    ROUND(100.0 * SUM(is_on_time) / COUNT(*), 2) AS on_time_percentage,
    ROUND(AVG(delay_days), 2) AS avg_delay_days
FROM delivery_operations_analytics
GROUP BY shipment_dayofweek
ORDER BY 
    CASE shipment_dayofweek
        WHEN 'Monday' THEN 1
        WHEN 'Tuesday' THEN 2
        WHEN 'Wednesday' THEN 3
        WHEN 'Thursday' THEN 4
        WHEN 'Friday' THEN 5
        WHEN 'Saturday' THEN 6
        WHEN 'Sunday' THEN 7
    END;
