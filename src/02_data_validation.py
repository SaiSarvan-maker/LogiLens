import pandas as pd

# Load raw data
df = pd.read_csv("data/raw/logistics_shipments_dataset.csv")

print("=" * 60)
print("LOGILENS - DATA VALIDATION")
print("=" * 60)

# --------------------------------------------------
# 1. Convert dates temporarily for validation
# --------------------------------------------------

df["Shipment_Date"] = pd.to_datetime(df["Shipment_Date"], errors="coerce")
df["Delivery_Date"] = pd.to_datetime(df["Delivery_Date"], errors="coerce")

print("\n1. DATE VALIDATION")
print("-" * 60)

print("Invalid Shipment_Date values:",
      df["Shipment_Date"].isna().sum())

print("Invalid Delivery_Date values:",
      df["Delivery_Date"].isna().sum())


# --------------------------------------------------
# 2. Check missing Delivery_Date against Status
# --------------------------------------------------

print("\n2. MISSING DELIVERY DATE vs STATUS")
print("-" * 60)

missing_delivery = df[df["Delivery_Date"].isna()]

print(
    missing_delivery["Status"]
    .value_counts()
    .to_string()
)


# --------------------------------------------------
# 3. Check missing Cost against Status
# --------------------------------------------------

print("\n3. MISSING COST vs STATUS")
print("-" * 60)

missing_cost = df[df["Cost"].isna()]

print(
    missing_cost["Status"]
    .value_counts()
    .to_string()
)


# --------------------------------------------------
# 4. Check date order
# --------------------------------------------------

print("\n4. DATE ORDER VALIDATION")
print("-" * 60)

invalid_date_order = df[
    df["Delivery_Date"].notna()
    & (df["Delivery_Date"] < df["Shipment_Date"])
]

print("Delivery before shipment:", len(invalid_date_order))


# --------------------------------------------------
# 5. Calculate actual transit days from dates
# --------------------------------------------------

df["Calculated_Transit_Days"] = (
    df["Delivery_Date"] - df["Shipment_Date"]
).dt.days

print("\n5. TRANSIT DAY CONSISTENCY")
print("-" * 60)

comparison = df[
    df["Delivery_Date"].notna()
    & df["Transit_Days"].notna()
].copy()

comparison["Transit_Days_Match"] = (
    comparison["Calculated_Transit_Days"]
    == comparison["Transit_Days"]
)

print(
    "Records where Transit_Days matches calculated dates:",
    comparison["Transit_Days_Match"].sum()
)

print(
    "Records where Transit_Days does NOT match:",
    (~comparison["Transit_Days_Match"]).sum()
)


# --------------------------------------------------
# 6. Check impossible numerical values
# --------------------------------------------------

print("\n6. NUMERICAL VALIDATION")
print("-" * 60)

print("Weight <= 0:",
      (df["Weight_kg"] <= 0).sum())

print("Cost <= 0:",
      (df["Cost"] <= 0).sum())

print("Distance <= 0:",
      (df["Distance_miles"] <= 0).sum())

print("Transit_Days <= 0:",
      (df["Transit_Days"] <= 0).sum())


# --------------------------------------------------
# 7. Status distribution
# --------------------------------------------------

print("\n7. STATUS DISTRIBUTION")
print("-" * 60)

print(df["Status"].value_counts().to_string())


# --------------------------------------------------
# 8. Validation summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)