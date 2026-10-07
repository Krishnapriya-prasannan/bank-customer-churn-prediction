import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve
)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

DATA_PATH = "European_Bank.csv"
ARTIFACT_DIR = "model_artifacts"

def load_data(filepath):
    print(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)
    return df

def engineer_features(df):
    """
    Cleans data and creates domain-specific banking features.
    """
    df_clean = df.copy()
    
    cols_to_drop = [col for col in ['CustomerId', 'Surname', 'Year'] if col in df_clean.columns]
    if cols_to_drop:
        df_clean = df_clean.drop(columns=cols_to_drop)
        
    if df_clean.isnull().sum().sum() > 0:
        df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
        
    # Domain Derived Features
    df_clean['Balance_Salary_Ratio'] = df_clean['Balance'] / (df_clean['EstimatedSalary'] + 1.0)
    df_clean['Product_Density'] = df_clean['NumOfProducts'] / (df_clean['Tenure'] + 1.0)
    df_clean['Engagement_Product_Interaction'] = df_clean['IsActiveMember'] * df_clean['NumOfProducts']
    df_clean['Tenure_Age_Ratio'] = df_clean['Tenure'] / (df_clean['Age'] + 1.0)
    df_clean['Credit_Age_Ratio'] = df_clean['CreditScore'] / (df_clean['Age'] + 1.0)
    df_clean['Is_Zero_Balance'] = (df_clean['Balance'] == 0).astype(int)
    df_clean['High_Risk_Age_Group'] = ((df_clean['Age'] >= 38) & (df_clean['Age'] <= 60)).astype(int)
    
    # Categorical Encoding
    df_encoded = pd.get_dummies(df_clean, columns=['Geography', 'Gender'], drop_first=False)
    
    for c in df_encoded.select_dtypes(include=['bool']).columns:
        df_encoded[c] = df_encoded[c].astype(int)
        
    return df_encoded

def optimize_decision_threshold(y_true, y_probs):
    """
    Finds decision threshold that maximizes F1-Score.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    best_f1 = float(f1_scores[best_idx])
    return round(best_threshold, 4), round(best_f1, 4)

def build_and_evaluate_models(X_train, X_test, y_train, y_test, feature_names):
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names, index=X_test.index)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=10, class_weight='balanced', random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.08, random_state=42)
    }
    
    if HAS_XGB:
        scale_pos = (len(y_train) - sum(y_train)) / sum(y_train)
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.08,
            scale_pos_weight=scale_pos, random_state=42, eval_metric='logloss'
        )
        
    results = {}
    fitted_models = {}
    curves_data = {}
    
    print("\n--- Training Candidate Models & Calculating Optimal Thresholds ---")
    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            y_pred_default = model.predict(X_train_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred_default = model.predict(X_train)
            y_prob = model.predict_proba(X_test)[:, 1]
            
        opt_thresh, opt_f1 = optimize_decision_threshold(y_test, y_prob)
        y_pred_opt = (y_prob >= opt_thresh).astype(int)
        
        acc = accuracy_score(y_test, y_pred_opt)
        prec = precision_score(y_test, y_pred_opt, zero_division=0)
        rec = recall_score(y_test, y_pred_opt, zero_division=0)
        f1 = f1_score(y_test, y_pred_opt, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred_opt).tolist()
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        precision_pts, recall_pts, thresh_pts = precision_recall_curve(y_test, y_prob)
        
        results[name] = {
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1-Score": round(float(f1), 4),
            "ROC-AUC": round(float(auc), 4),
            "OptimalThreshold": opt_thresh,
            "ConfusionMatrix": cm
        }
        
        curves_data[name] = {
            "fpr": [round(float(val), 4) for val in fpr],
            "tpr": [round(float(val), 4) for val in tpr],
            "precision": [round(float(val), 4) for val in precision_pts],
            "recall": [round(float(val), 4) for val in recall_pts]
        }
        
        fitted_models[name] = model
        print(f"Model: {name:20s} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f} | Opt Thresh: {opt_thresh}")
        
    return scaler, fitted_models, results, curves_data

def compute_shap_and_importances(best_model, X_train, feature_names):
    feature_importances = {}
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        feature_importances = dict(zip(feature_names, [round(float(i), 5) for i in importances]))
        feature_importances = dict(sorted(feature_importances.items(), key=lambda x: x[1], reverse=True))
    elif hasattr(best_model, 'coef_'):
        importances = np.abs(best_model.coef_[0])
        feature_importances = dict(zip(feature_names, [round(float(i), 5) for i in importances]))
        feature_importances = dict(sorted(feature_importances.items(), key=lambda x: x[1], reverse=True))
        
    shap_summary = {}
    if HAS_SHAP:
        try:
            print("Calculating SHAP Explainer...")
            explainer = shap.Explainer(best_model, X_train)
            sample_X = X_train.sample(min(500, len(X_train)), random_state=42)
            shap_values = explainer(sample_X)
            
            if len(shap_values.values.shape) == 3:
                vals = np.abs(shap_values.values[:, :, 1]).mean(0)
            else:
                vals = np.abs(shap_values.values).mean(0)
                
            shap_summary = dict(zip(feature_names, [round(float(v), 5) for v in vals]))
            shap_summary = dict(sorted(shap_summary.items(), key=lambda x: x[1], reverse=True))
        except Exception as e:
            print(f"SHAP note: {e}")
            shap_summary = feature_importances
    else:
        shap_summary = feature_importances
        
    return feature_importances, shap_summary

def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    df = load_data(DATA_PATH)
    
    df_encoded = engineer_features(df)
    X = df_encoded.drop(columns=['Exited'])
    y = df_encoded['Exited']
    feature_names = list(X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    scaler, fitted_models, metrics_dict, curves_data = build_and_evaluate_models(
        X_train, X_test, y_train, y_test, feature_names
    )
    
    best_model_name = max(metrics_dict.items(), key=lambda item: item[1]['ROC-AUC'])[0]
    best_model = fitted_models[best_model_name]
    print(f"\nSelected Best Model: {best_model_name}")
    
    feature_importances, shap_summary = compute_shap_and_importances(best_model, X_train, feature_names)
    
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
            "churn_rate": round(float(y.mean()), 4),
            "avg_balance": round(float(df['Balance'].mean()), 2),
            "total_balance_at_risk": round(float(df[df['Exited']==1]['Balance'].sum()), 2)
        }
    }
    
    with open(os.path.join(ARTIFACT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    print("Enhanced model training pipeline complete!")

if __name__ == "__main__":
    main()
