import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve
)

# Optional XGBoost import with fallback
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# Optional SHAP import
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

DATA_PATH = "European_Bank.csv"
ARTIFACT_DIR = "model_artifacts"

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    return df

def engineer_features(df):
    """
    Performs data cleaning and derived feature engineering.
    """
    df_clean = df.copy()
    
    # Drop non-informative features if present
    cols_to_drop = [col for col in ['CustomerId', 'Surname', 'Year'] if col in df_clean.columns]
    if cols_to_drop:
        df_clean = df_clean.drop(columns=cols_to_drop)
        print(f"Dropped non-informative columns: {cols_to_drop}")
    
    # Handle missing values if any
    if df_clean.isnull().sum().sum() > 0:
        print("Handling missing values...")
        df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
    
    # Feature Engineering
    # 1. Balance to Estimated Salary ratio
    df_clean['Balance_Salary_Ratio'] = df_clean['Balance'] / (df_clean['EstimatedSalary'] + 1.0)
    
    # 2. Product Density (Products per year of tenure)
    df_clean['Product_Density'] = df_clean['NumOfProducts'] / (df_clean['Tenure'] + 1.0)
    
    # 3. Engagement-Product Interaction
    df_clean['Engagement_Product_Interaction'] = df_clean['IsActiveMember'] * df_clean['NumOfProducts']
    
    # 4. Tenure-to-Age Ratio
    df_clean['Tenure_Age_Ratio'] = df_clean['Tenure'] / (df_clean['Age'] + 1.0)
    
    # 5. Credit Score to Age Ratio
    df_clean['Credit_Age_Ratio'] = df_clean['CreditScore'] / (df_clean['Age'] + 1.0)
    
    # 6. Zero Balance Binary Indicator
    df_clean['Is_Zero_Balance'] = (df_clean['Balance'] == 0).astype(int)
    
    # 7. High Risk Age Window (38-60 age demographic has higher historical churn)
    df_clean['High_Risk_Age_Group'] = ((df_clean['Age'] >= 38) & (df_clean['Age'] <= 60)).astype(int)
    
    # One-Hot Encoding for categorical variables: Geography and Gender
    categorical_cols = ['Geography', 'Gender']
    df_encoded = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=False)
    
    # Convert bool columns to int
    bool_cols = df_encoded.select_dtypes(include=['bool']).columns
    for c in bool_cols:
        df_encoded[c] = df_encoded[c].astype(int)
        
    print(f"Engineered dataset features: {df_encoded.columns.tolist()}")
    return df_encoded

def build_and_evaluate_models(X_train, X_test, y_train, y_test, feature_names):
    """
    Trains candidate ML models and evaluates benchmark metrics.
    """
    # Scale numerical features
    scaler = StandardScaler()
    
    # Scale X_train and X_test
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.fit_transform(X_test), columns=feature_names, index=X_test.index)
    
    # Define models dictionary
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    }
    
    if HAS_XGB:
        # Scale pos weight ratio for balanced churn learning
        scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss'
        )
    else:
        models["HistGradientBoosting"] = HistGradientBoostingClassifier(max_depth=6, random_state=42)

    results = {}
    fitted_models = {}
    curves_data = {}
    
    print("\n--- Model Training & Benchmark Evaluation ---")
    for name, model in models.items():
        # Tree models use unscaled or scaled features (scaled is safe for all)
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        # Calculate ROC and Precision-Recall curves
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        precision_pts, recall_pts, _ = precision_recall_curve(y_test, y_prob)
        
        results[name] = {
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1-Score": round(float(f1), 4),
            "ROC-AUC": round(float(auc), 4),
            "ConfusionMatrix": cm
        }
        
        curves_data[name] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "precision": precision_pts.tolist(),
            "recall": recall_pts.tolist()
        }
        
        fitted_models[name] = model
        print(f"Model: {name:20s} | Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
        
    return scaler, fitted_models, results, curves_data

def compute_shap_and_importances(best_model, X_train, feature_names):
    """
    Computes global feature importances and SHAP values.
    """
    feature_importances = {}
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        feature_importances = dict(zip(feature_names, [round(float(i), 5) for i in importances]))
        # Sort descending
        feature_importances = dict(sorted(feature_importances.items(), key=lambda x: x[1], reverse=True))
    elif hasattr(best_model, 'coef_'):
        importances = np.abs(best_model.coef_[0])
        feature_importances = dict(zip(feature_names, [round(float(i), 5) for i in importances]))
        feature_importances = dict(sorted(feature_importances.items(), key=lambda x: x[1], reverse=True))
        
    shap_summary = {}
    if HAS_SHAP:
        try:
            print("Calculating SHAP values...")
            explainer = shap.Explainer(best_model, X_train)
            sample_X = X_train.sample(min(500, len(X_train)), random_state=42)
            shap_values = explainer(sample_X)
            
            # Mean absolute SHAP value per feature
            if len(shap_values.values.shape) == 3:
                vals = np.abs(shap_values.values[:, :, 1]).mean(0)
            else:
                vals = np.abs(shap_values.values).mean(0)
                
            shap_summary = dict(zip(feature_names, [round(float(v), 5) for v in vals]))
            shap_summary = dict(sorted(shap_summary.items(), key=lambda x: x[1], reverse=True))
        except Exception as e:
            print(f"SHAP calculation note: {e}")
            shap_summary = feature_importances
    else:
        shap_summary = feature_importances
        
    return feature_importances, shap_summary

def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    df = load_data(DATA_PATH)
    
    # Process & engineer features
    df_encoded = engineer_features(df)
    
    X = df_encoded.drop(columns=['Exited'])
    y = df_encoded['Exited']
    feature_names = list(X.columns)
    
    # Stratified Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nTrain set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")
    print(f"Train Churn Rate: {y_train.mean():.4f} | Test Churn Rate: {y_test.mean():.4f}")
    
    # Build models & evaluate
    scaler, fitted_models, metrics_dict, curves_data = build_and_evaluate_models(
        X_train, X_test, y_train, y_test, feature_names
    )
    
    # Select Best Model based on ROC-AUC / F1
    best_model_name = max(metrics_dict.items(), key=lambda item: item[1]['ROC-AUC'])[0]
    print(f"\nBest Model selected based on ROC-AUC: {best_model_name}")
    best_model = fitted_models[best_model_name]
    
    # Feature importances & SHAP
    feature_importances, shap_summary = compute_shap_and_importances(best_model, X_train, feature_names)
    
    # Save artifacts
    print(f"\nSaving model artifacts to {ARTIFACT_DIR}/...")
    joblib.dump(best_model, os.path.join(ARTIFACT_DIR, "best_churn_model.pkl"))
    joblib.dump(fitted_models, os.path.join(ARTIFACT_DIR, "all_candidate_models.pkl"))
    joblib.dump(scaler, os.path.join(ARTIFACT_DIR, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(ARTIFACT_DIR, "feature_names.pkl"))
    
    metadata = {
        "best_model_name": best_model_name,
        "feature_names": feature_names,
        "metrics": metrics_dict,
        "feature_importances": feature_importances,
        "shap_summary": shap_summary,
        "curves_data": curves_data,
        "dataset_stats": {
            "total_records": len(df),
            "churn_records": int(y.sum()),
            "retained_records": int((y == 0).sum()),
            "churn_rate": round(float(y.mean()), 4)
        }
    }
    
    with open(os.path.join(ARTIFACT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    print("Model pipeline training & artifact export successfully completed!")

if __name__ == "__main__":
    main()
