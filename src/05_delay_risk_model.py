"""
LogiLens - Machine Learning Delay Risk Estimator
------------------------------------------------
Fulfills Step 5 of the Delivery Delay & Operations Analytics project:
1. Builds an interpretable Decision Tree Classifier (and benchmark models)
   to predict the likelihood of delivery delay given shipment attributes known prior to delivery.
2. Target Definition (Defensible & Leakage-Free):
   target = 1 if Delivery_Status == "Delayed", 0 if Delivery_Status == "Delivered".
3. Evaluates performance (ROC-AUC, Precision, Recall, F1, Confusion Matrix).
4. Serializes the trained pipeline and exposes a live inference function.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def load_modeling_data(file_path: str = "data/processed/delivery_operations_analytics.csv"):
    """
    Loads and filters dataset for model training.
    Defensively enforces no data leakage: outcome dates, variances, and delay flags are excluded.
    """
    df = pd.read_csv(file_path)
    
    # Filter resolved shipments (Delivered or Delayed) for defensible binary classification
    status_col = "Delivery_Status" if "Delivery_Status" in df.columns else "Status"
    resolved = df[df[status_col].isin(["Delivered", "Delayed"])].copy()
    
    # Primary Target: 1 if Status == Delayed, else 0
    y = (resolved[status_col] == "Delayed").astype(int).values
    
    # Feature set strictly restricted to pre-dispatch / shipment creation attributes
    feature_cols = [
        "Origin_Warehouse",
        "Carrier",
        "Destination",
        "Product_Type",
        "Priority",
        "Shipment_DayOfWeek",
        "Season",
        "Distance_Tier",
        "Distance_miles",
        "Weight_kg",
        "Cost",
        "Promised_SLA_Days"
    ]
    
    X = resolved[feature_cols].copy()
    
    categorical_features = [
        "Origin_Warehouse",
        "Carrier",
        "Destination",
        "Product_Type",
        "Priority",
        "Shipment_DayOfWeek",
        "Season",
        "Distance_Tier"
    ]
    
    numerical_features = [
        "Distance_miles",
        "Weight_kg",
        "Cost",
        "Promised_SLA_Days"
    ]
    
    return X, y, categorical_features, numerical_features


def build_preprocessor(categorical_features, numerical_features):
    """Creates scikit-learn ColumnTransformer for preprocessing."""
    cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    num_transformer = StandardScaler()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_transformer, categorical_features),
            ("num", num_transformer, numerical_features)
        ]
    )
    return preprocessor


def train_and_evaluate_models(X, y, categorical_features, numerical_features):
    """
    Trains Decision Tree, Logistic Regression, and Random Forest.
    Returns the primary interpretable Decision Tree pipeline and metrics.
    """
    print("\n" + "=" * 65)
    print("STEP 5: MACHINE LEARNING DELAY RISK ESTIMATION (LEAKAGE-FREE)")
    print("=" * 65)
    
    # 80/20 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"Dataset split: Train = {len(X_train)} samples, Test = {len(X_test)} samples")
    print(f"Delay rate in train: {np.mean(y_train)*100:.2f}%, in test: {np.mean(y_test)*100:.2f}%")
    
    preprocessor = build_preprocessor(categorical_features, numerical_features)
    
    # -------------------------------------------------------------
    # 1. Primary Model: Interpretable Decision Tree Classifier
    # -------------------------------------------------------------
    # Max depth 4 keeps the tree explainable to operations executives
    tree_clf = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42
    )
    tree_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", tree_clf)
    ])
    
    tree_pipeline.fit(X_train, y_train)
    y_pred_tree = tree_pipeline.predict(X_test)
    y_prob_tree = tree_pipeline.predict_proba(X_test)[:, 1]
    
    tree_acc = accuracy_score(y_test, y_pred_tree)
    tree_prec = precision_score(y_test, y_pred_tree, zero_division=0)
    tree_rec = recall_score(y_test, y_pred_tree, zero_division=0)
    tree_f1 = f1_score(y_test, y_pred_tree, zero_division=0)
    tree_auc = roc_auc_score(y_test, y_prob_tree)
    
    print("\n--- 1. Primary Decision Tree Performance (Depth=4) ---")
    print(f"Accuracy:  {tree_acc:.4f}")
    print(f"Precision: {tree_prec:.4f}")
    print(f"Recall:    {tree_rec:.4f}")
    print(f"F1-Score:  {tree_f1:.4f}")
    print(f"ROC-AUC:   {tree_auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred_tree, target_names=["On-Time", "Delayed"]))
    
    # Cross Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(tree_pipeline, X, y, cv=cv, scoring="roc_auc")
    print(f"5-Fold Cross-Validation ROC-AUC: {cv_auc.mean():.4f} (+/- {cv_auc.std():.4f})")
    
    # -------------------------------------------------------------
    # 2. Benchmark Comparison: Logistic Regression & Random Forest
    # -------------------------------------------------------------
    lr_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_auc = roc_auc_score(y_test, lr_pipeline.predict_proba(X_test)[:, 1])
    
    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=42))
    ])
    rf_pipeline.fit(X_train, y_train)
    rf_auc = roc_auc_score(y_test, rf_pipeline.predict_proba(X_test)[:, 1])
    
    print("\n--- Model Benchmark Summary (ROC-AUC Comparison) ---")
    print(f"Decision Tree Classifier:   ROC-AUC = {tree_auc:.4f}")
    print(f"Logistic Regression:        ROC-AUC = {lr_auc:.4f}")
    print(f"Random Forest Classifier:   ROC-AUC = {rf_auc:.4f}")
    print("Note: Honest evaluation indicates pre-dispatch features offer moderate predictive signal for delay status.")
    
    # -------------------------------------------------------------
    # 3. Extract Feature Names & Importances
    # -------------------------------------------------------------
    ohe = tree_pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    encoded_cat_names = list(ohe.get_feature_names_out(categorical_features))
    all_feature_names = encoded_cat_names + numerical_features
    
    importances = tree_clf.feature_importances_
    feat_imp_df = pd.DataFrame({
        "Feature": all_feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False)
    
    top_features = feat_imp_df[feat_imp_df["Importance"] > 0].copy()
    print("\n--- Key Operational Risk Drivers (Decision Tree Feature Importances) ---")
    print(top_features.head(10).to_string(index=False))
    
    # -------------------------------------------------------------
    # 4. Extract Text Decision Rules
    # -------------------------------------------------------------
    tree_rules = export_text(tree_clf, feature_names=all_feature_names, max_depth=3)
    print("\n--- Core Operational Decision Rules (Top 3 Levels) ---")
    print(tree_rules[:800] + "...")
    
    metrics = {
        "model_type": "DecisionTreeClassifier",
        "max_depth": 4,
        "accuracy": round(tree_acc, 4),
        "precision": round(tree_prec, 4),
        "recall": round(tree_rec, 4),
        "f1_score": round(tree_f1, 4),
        "roc_auc": round(tree_auc, 4),
        "cv_roc_auc_mean": round(cv_auc.mean(), 4),
        "cv_roc_auc_std": round(cv_auc.std(), 4),
        "benchmark_comparison": {
            "Decision_Tree_AUC": round(tree_auc, 4),
            "Logistic_Regression_AUC": round(lr_auc, 4),
            "Random_Forest_AUC": round(rf_auc, 4)
        },
        "top_features": top_features.head(8).to_dict(orient="records")
    }
    
    return tree_pipeline, metrics, top_features, all_feature_names, X_test, y_test


def plot_model_diagnostics(tree_pipeline, top_features, X_test, y_test, output_dir: str = "reports/figures"):
    """Saves visual model diagnostics (Feature Importances, Tree Structure, Confusion Matrix)."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Feature Importances Plot
    plt.figure(figsize=(10, 5))
    top_plot = top_features.head(10).sort_values("Importance", ascending=True)
    plt.barh(top_plot["Feature"], top_plot["Importance"], color="#337ab7", alpha=0.9)
    plt.title("Top Operational Delay Risk Predictors (Decision Tree Gini Importance)")
    plt.xlabel("Importance Weight")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importances.png"), dpi=300)
    plt.close()
    
    # 2. Confusion Matrix Plot
    y_pred = tree_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Pred On-Time", "Pred Delayed"],
        yticklabels=["Actual On-Time", "Actual Delayed"]
    )
    plt.title("Confusion Matrix: Delivery Delay Prediction")
    plt.ylabel("Actual Ground Truth")
    plt.xlabel("Model Prediction")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close()
    print("-> Model diagnostic plots saved to reports/figures/")


def save_model_artifacts(tree_pipeline, metrics, all_feature_names, output_dir: str = "models"):
    """Serializes model pipeline and metadata."""
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "delay_risk_model.joblib")
    metrics_path = os.path.join(output_dir, "model_metrics.json")
    features_path = os.path.join(output_dir, "feature_names.json")
    
    joblib.dump(tree_pipeline, model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    with open(features_path, "w") as f:
        json.dump(all_feature_names, f, indent=4)
        
    print(f"-> Serialized model pipeline to: {model_path}")
    print(f"-> Serialized model metrics to:  {metrics_path}")


def predict_delay_risk(shipment_data: dict, model_path: str = "models/delay_risk_model.joblib") -> dict:
    """
    Operational inference utility function for live predictions.
    Accepts a dictionary of shipment features and returns risk assessment.
    """
    pipeline = joblib.load(model_path)
    
    # Calculate derived features if not present
    distance = shipment_data.get("Distance_miles", 1000)
    if "Distance_Tier" not in shipment_data:
        if distance < 500:
            shipment_data["Distance_Tier"] = "Short-Haul (<500 mi)"
        elif distance <= 1500:
            shipment_data["Distance_Tier"] = "Medium-Haul (500-1500 mi)"
        else:
            shipment_data["Distance_Tier"] = "Long-Haul (>1500 mi)"
            
    df_input = pd.DataFrame([shipment_data])
    prob_delay = float(pipeline.predict_proba(df_input)[0, 1])
    is_delayed = int(prob_delay >= 0.50)
    
    if prob_delay < 0.20:
        risk_tier = "Low Risk"
        recommendation = "Route operates within standard SLA safety bounds. Standard dispatch schedule."
    elif prob_delay < 0.45:
        risk_tier = "Moderate Risk"
        recommendation = "Moderate bottleneck risk detected. Consider dispatching in early morning wave."
    else:
        risk_tier = "High Risk"
        recommendation = "High risk of SLA breach. Recommend priority routing, alternative carrier, or buffer SLA."
        
    return {
        "delay_probability": round(prob_delay, 3),
        "delay_probability_pct": f"{prob_delay*100:.1f}%",
        "predicted_status": "Delayed" if is_delayed == 1 else "On-Time",
        "risk_tier": risk_tier,
        "recommendation": recommendation
    }


def run_training_pipeline():
    """Main training orchestrator."""
    X, y, cat_features, num_features = load_modeling_data()
    tree_pipeline, metrics, top_features, all_feature_names, X_test, y_test = train_and_evaluate_models(
        X, y, cat_features, num_features
    )
    plot_model_diagnostics(tree_pipeline, top_features, X_test, y_test)
    save_model_artifacts(tree_pipeline, metrics, all_feature_names)
    
    # Test sample inference
    sample_shipment = {
        "Origin_Warehouse": "Warehouse_MIA",
        "Carrier": "Amazon Logistics",
        "Destination": "Boston",
        "Product_Type": "Perishables & Grocery",
        "Priority": "Standard",
        "Shipment_DayOfWeek": "Tuesday",
        "Season": "Winter",
        "Distance_Tier": "Long-Haul (>1500 mi)",
        "Distance_miles": 1780,
        "Weight_kg": 24.5,
        "Cost": 220.0,
        "Promised_SLA_Days": 3
    }
    pred_result = predict_delay_risk(sample_shipment)
    print("\n--- Test Live Shipment Risk Inference ---")
    print("Input Shipment:", sample_shipment)
    print("Prediction Result:", json.dumps(pred_result, indent=2))


if __name__ == "__main__":
    run_training_pipeline()

