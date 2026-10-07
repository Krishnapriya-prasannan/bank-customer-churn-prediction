import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Set Page Config
st.set_page_config(
    page_title="Enterprise Churn Intelligence Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Enterprise Aesthetics
st.markdown("""
<style>
    :root {
        --bg-main: #0b0f19;
        --card-bg: #151c2c;
        --card-border: #26324a;
        --text-primary: #ffffff;
        --text-secondary: #90a4ae;
        --accent-blue: #00b0ff;
        --accent-green: #00e676;
        --accent-yellow: #ffb300;
        --accent-red: #ff1744;
    }
    
    .stApp {
        background-color: var(--bg-main);
        color: var(--text-primary);
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(21,28,44,0.95), rgba(30,41,64,0.85));
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.4);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: var(--accent-blue);
    }
    .metric-val {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 5px 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    
    .risk-badge-low {
        background-color: rgba(0, 230, 118, 0.15);
        color: #00e676;
        border: 1px solid #00e676;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
    }
    .risk-badge-medium {
        background-color: rgba(255, 179, 0, 0.15);
        color: #ffb300;
        border: 1px solid #ffb300;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
    }
    .risk-badge-high {
        background-color: rgba(255, 23, 68, 0.15);
        color: #ff1744;
        border: 1px solid #ff1744;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

ARTIFACT_DIR = "model_artifacts"
DATA_PATH = "European_Bank.csv"

@st.cache_resource
def load_artifacts():
    metadata_path = os.path.join(ARTIFACT_DIR, "metadata.json")
    if not os.path.exists(metadata_path):
        st.error("Model artifacts not found! Run `python train_model.py` first.")
        st.stop()
        
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
        
    best_model = joblib.load(os.path.join(ARTIFACT_DIR, "best_churn_model.pkl"))
    candidate_models = joblib.load(os.path.join(ARTIFACT_DIR, "all_candidate_models.pkl"))
    scaler = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(ARTIFACT_DIR, "feature_names.pkl"))
    
    df_raw = pd.read_csv(DATA_PATH)
    
    return metadata, best_model, candidate_models, scaler, feature_names, df_raw

metadata, best_model, candidate_models, scaler, feature_names, df_raw = load_artifacts()

def preprocess_dataframe_batch(df):
    """
    Transforms any raw input dataframe into the engineered feature matrix.
    """
    df_clean = df.copy()
    
    # Store ID columns for output if present
    id_cols = {}
    if 'CustomerId' in df_clean.columns:
        id_cols['CustomerId'] = df_clean['CustomerId']
    if 'Surname' in df_clean.columns:
        id_cols['Surname'] = df_clean['Surname']
        
    cols_to_drop = [c for c in ['CustomerId', 'Surname', 'Year', 'Exited'] if c in df_clean.columns]
    df_feat = df_clean.drop(columns=cols_to_drop)
    
    # Fill missing values if any
    df_feat = df_feat.fillna(df_feat.median(numeric_only=True))
    
    # Derived Features
    df_feat['Balance_Salary_Ratio'] = df_feat['Balance'] / (df_feat['EstimatedSalary'] + 1.0)
    df_feat['Product_Density'] = df_feat['NumOfProducts'] / (df_feat['Tenure'] + 1.0)
    df_feat['Engagement_Product_Interaction'] = df_feat['IsActiveMember'] * df_feat['NumOfProducts']
    df_feat['Tenure_Age_Ratio'] = df_feat['Tenure'] / (df_feat['Age'] + 1.0)
    df_feat['Credit_Age_Ratio'] = df_feat['CreditScore'] / (df_feat['Age'] + 1.0)
    df_feat['Is_Zero_Balance'] = (df_feat['Balance'] == 0).astype(int)
    df_feat['High_Risk_Age_Group'] = ((df_feat['Age'] >= 38) & (df_feat['Age'] <= 60)).astype(int)
    
    # One-Hot Encoding
    if 'Geography' in df_feat.columns:
        df_feat['Geography_France'] = (df_feat['Geography'] == 'France').astype(int)
        df_feat['Geography_Germany'] = (df_feat['Geography'] == 'Germany').astype(int)
        df_feat['Geography_Spain'] = (df_feat['Geography'] == 'Spain').astype(int)
        df_feat = df_feat.drop(columns=['Geography'])
        
    if 'Gender' in df_feat.columns:
        df_feat['Gender_Female'] = (df_feat['Gender'] == 'Female').astype(int)
        df_feat['Gender_Male'] = (df_feat['Gender'] == 'Male').astype(int)
        df_feat = df_feat.drop(columns=['Gender'])
        
    # Ensure missing columns (if any) are added with 0
    for col in feature_names:
        if col not in df_feat.columns:
            df_feat[col] = 0
            
    df_feat = df_feat[feature_names]
    return df_feat, id_cols

def assign_retention_playbook(row):
    if row['IsActiveMember'] == 0:
        return "Digital App Engagement Incentive (0.5% Deposit Bonus)"
    elif row['NumOfProducts'] == 1:
        return "2nd Product Cross-Sell Offer (Zero-Fee Credit Card)"
    elif row['NumOfProducts'] >= 3:
        return "Relationship Manager Call (Product Fee Rationalization)"
    elif row['Geography_Germany'] == 1:
        return "Germany Regional VIP Loyalty Program"
    elif row['High_Risk_Age_Group'] == 1:
        return "Personalized Wealth & Mortgage Refinancing Consultation"
    elif row['Is_Zero_Balance'] == 1:
        return "Direct Deposit Salary Bonus Campaign"
    else:
        return "Standard Loyalty Nurturing"

# Sidebar Navigation
st.sidebar.title("🏦 Bank Churn Intelligence")
st.sidebar.markdown("Enterprise Predictive Analytics Platform")

module = st.sidebar.radio(
    "Select Enterprise Module",
    [
        "📊 Portfolio Overview & Health",
        "📁 Batch Customer Scoring & CSV Export",
        "🧮 Individual Customer Risk Calculator",
        "💰 Financial ROI & Revenue Calculator",
        "⚡ Model Benchmarks & Decision Thresholds",
        "🔍 SHAP & Feature Explainability",
        "🧪 What-If Scenario Simulator"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption(f"**Primary Model:** {metadata['best_model_name']}")
st.sidebar.caption(f"**ROC-AUC Score:** {metadata['metrics'][metadata['best_model_name']]['ROC-AUC']:.4f}")
st.sidebar.caption("© European Central Bank AI Governance Compliant")

# ==========================================
# MODULE 1: PORTFOLIO OVERVIEW
# ==========================================
if module == "📊 Portfolio Overview & Health":
    st.title("📊 Retail Banking Portfolio Overview")
    st.markdown("Macro-level balance sheet liquidity, customer retention KPIs, and demographic distributions.")
    
    stats = metadata["dataset_stats"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Portfolio Customers</div>
            <div class="metric-val" style="color: #00b0ff;">{stats['total_records']:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Overall Churn Rate</div>
            <div class="metric-val" style="color: #ff1744;">{stats['churn_rate']*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average Account Balance</div>
            <div class="metric-val" style="color: #00e676;">€{stats['avg_balance']:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Balance at Risk</div>
            <div class="metric-val" style="color: #ffb300;">€{stats['total_balance_at_risk']:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Geographic Churn Exposure")
        geo_df = df_raw.groupby(['Geography', 'Exited']).size().reset_index(name='Count')
        geo_df['Status'] = geo_df['Exited'].map({0: 'Retained', 1: 'Churned'})
        fig_geo = px.bar(
            geo_df, x='Geography', y='Count', color='Status', barmode='group',
            color_discrete_map={'Retained': '#00e676', 'Churned': '#ff1744'},
            title="Customer Count by Geography"
        )
        fig_geo.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#ffffff')
        st.plotly_chart(fig_geo, use_container_width=True)
        
    with col2:
        st.subheader("The Product Paradox (Churn Rate by Products Owned)")
        prod_df = df_raw.groupby('NumOfProducts')['Exited'].agg(['count', 'mean']).reset_index()
        prod_df['ChurnRate%'] = prod_df['mean'] * 100
        fig_prod = px.bar(
            prod_df, x='NumOfProducts', y='ChurnRate%', text='ChurnRate%',
            color='ChurnRate%', color_continuous_scale='Reds',
            title="Churn Probability by Product Count"
        )
        fig_prod.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_prod.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#ffffff')
        st.plotly_chart(fig_prod, use_container_width=True)

# ==========================================
# MODULE 2: BATCH CUSTOMER SCORING & CSV EXPORT
# ==========================================
elif module == "📁 Batch Customer Scoring & CSV Export":
    st.title("📁 Batch Customer Risk Scoring & CRM Export")
    st.markdown("Upload any bank customer dataset (CSV format) to batch calculate churn probabilities, assign risk tiers, and generate automated CRM retention action plans.")
    
    uploaded_file = st.file_uploader("Upload Customer Dataset (CSV)", type=["csv"])
    
    col_sample1, col_sample2 = st.columns([1, 3])
    with col_sample1:
        # Download Sample Template
        sample_df = df_raw.head(50).drop(columns=['Exited'])
        csv_sample = sample_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Sample Test CSV", data=csv_sample, file_name="sample_bank_customers.csv", mime="text/csv")
        
    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)
        st.success(f"Successfully loaded dataset with {len(df_input):,} customer records.")
        
        with st.spinner("Processing batch feature engineering & running predictions..."):
            df_feat, id_cols = preprocess_dataframe_batch(df_input)
            probs = best_model.predict_proba(df_feat)[:, 1]
            
            df_result = df_input.copy()
            df_result['Churn_Probability_%'] = np.round(probs * 100, 2)
            df_result['Risk_Tier'] = pd.cut(
                probs, bins=[-0.01, 0.30, 0.60, 1.01],
                labels=['Low Risk', 'Medium Risk', 'High Risk']
            )
            
            df_result['Retention_Action_Playbook'] = df_feat.apply(assign_retention_playbook, axis=1)
            
        st.markdown("---")
        st.subheader("Batch Prediction Summary")
        
        b1, b2, b3, b4 = st.columns(4)
        total_scored = len(df_result)
        high_risk = (df_result['Risk_Tier'] == 'High Risk').sum()
        med_risk = (df_result['Risk_Tier'] == 'Medium Risk').sum()
        low_risk = (df_result['Risk_Tier'] == 'Low Risk').sum()
        
        with b1:
            st.metric("Total Scored Customers", f"{total_scored:,}")
        with b2:
            st.metric("High Risk Customers (>60%)", f"{high_risk:,}", delta=f"{high_risk/total_scored*100:.1f}%", delta_color="inverse")
        with b3:
            st.metric("Medium Risk Customers (30-60%)", f"{med_risk:,}")
        with b4:
            st.metric("Low Risk Customers (<30%)", f"{low_risk:,}")
            
        st.markdown("### Filter Scored Customer List")
        filter_tier = st.multiselect("Filter by Risk Tier", ["High Risk", "Medium Risk", "Low Risk"], default=["High Risk", "Medium Risk"])
        
        filtered_results = df_result[df_result['Risk_Tier'].isin(filter_tier)]
        st.dataframe(filtered_results, use_container_width=True)
        
        # Download Enriched CSV Button
        csv_download = df_result.to_csv(index=False).encode('utf-8')
        st.download_button(
            "🚀 Export Complete Scored Dataset with Action Playbooks (CSV)",
            data=csv_download,
            file_name="churn_risk_scored_customers.csv",
            mime="text/csv"
        )

# ==========================================
# MODULE 3: INDIVIDUAL CUSTOMER RISK CALCULATOR
# ==========================================
elif module == "🧮 Individual Customer Risk Calculator":
    st.title("🧮 Individual Customer Risk Calculator & SHAP Explanation")
    st.markdown("Input customer demographics and account details to generate a real-time risk score, risk factors, and custom retention playbook.")
    
    with st.form("single_risk_form"):
        st.subheader("Customer Demographics & Account Inputs")
        c1, c2, c3 = st.columns(3)
        with c1:
            credit_score = st.slider("Credit Score", 300, 850, 650)
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
            gender = st.selectbox("Gender", ["Female", "Male"])
            age = st.slider("Age", 18, 92, 42)
        with c2:
            tenure = st.slider("Tenure (Years)", 0, 10, 4)
            balance = st.number_input("Balance (€)", min_value=0.0, max_value=300000.0, value=85000.0, step=5000.0)
            salary = st.number_input("Estimated Salary (€)", min_value=0.0, max_value=250000.0, value=95000.0, step=5000.0)
        with c3:
            num_products = st.selectbox("Number of Products", [1, 2, 3, 4], index=0)
            has_card = st.selectbox("Has Credit Card?", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")
            is_active = st.selectbox("Is Active Member?", [1, 0], format_func=lambda x: "Yes (Active)" if x==1 else "No (Inactive)")
            
        submit_calc = st.form_submit_button("⚡ Predict Churn Probability")
        
    inputs = {
        'CreditScore': credit_score, 'Geography': geography, 'Gender': gender,
        'Age': age, 'Tenure': tenure, 'Balance': balance,
        'NumOfProducts': num_products, 'HasCrCard': has_card,
        'IsActiveMember': is_active, 'EstimatedSalary': salary
    }
    
    df_single = pd.DataFrame([inputs])
    df_feat, _ = preprocess_dataframe_batch(df_single)
    prob = best_model.predict_proba(df_feat)[0][1]
    
    st.markdown("---")
    r1, r2 = st.columns([1, 2])
    with r1:
        color = "#ff1744" if prob > 0.6 else ("#ffb300" if prob >= 0.3 else "#00e676")
        tier = "High Risk" if prob > 0.6 else ("Medium Risk" if prob >= 0.3 else "Low Risk")
        badge = "risk-badge-high" if prob > 0.6 else ("risk-badge-medium" if prob >= 0.3 else "risk-badge-low")
        st.markdown(f"""
        <div class="metric-card" style="padding: 30px;">
            <div class="metric-label">Predicted Churn Probability</div>
            <div class="metric-val" style="font-size: 3.2rem; color: {color};">{prob*100:.1f}%</div>
            <div class="{badge}">{tier}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with r2:
        st.markdown("#### Automated Action Playbook")
        playbook = assign_retention_playbook(df_feat.iloc[0])
        st.info(f"🎯 **Recommended Retention Campaign**: {playbook}")
        
        st.markdown("##### Key Risk Drivers:")
        if is_active == 0:
            st.markdown("- ⚠️ Inactive Member Status (High risk multiplier)")
        if num_products == 1:
            st.markdown("- ⚠️ Single product lock-in (Vulnerable to competitor offers)")
        elif num_products >= 3:
            st.markdown("- 🔴 Product overcrowding (High fee dissatisfaction risk)")
        if geography == "Germany":
            st.markdown("- 🌐 Germany regional demographic risk pool")
        if age >= 38 and age <= 60:
            st.markdown("- 👤 Customer is in peak churn age window (38–60)")

# ==========================================
# MODULE 4: FINANCIAL ROI CALCULATOR
# ==========================================
elif module == "💰 Financial ROI & Revenue Calculator":
    st.title("💰 Financial ROI & Loss Mitigation Calculator")
    st.markdown("Quantify balance sheet revenue loss from customer churn and calculate net ROI of targeted Machine Learning retention campaigns vs. blanket marketing.")
    
    f1, f2, f3 = st.columns(3)
    with f1:
        clv_annual = st.number_input("Average Customer Annual Revenue (€)", value=1200.0, step=100.0)
    with f2:
        offer_cost = st.number_input("Retention Offer Cost per Customer (€)", value=150.0, step=25.0)
    with f3:
        conversion_rate = st.slider("Campaign Retention Success Rate (%)", 10, 80, 40) / 100.0
        
    st.markdown("---")
    
    # Calculate portfolio revenue metrics
    stats = metadata["dataset_stats"]
    total_churners = stats["churn_records"]
    total_retained = stats["retained_records"]
    
    # ML Targeted Campaign Metrics
    best_name = metadata["best_model_name"]
    best_metrics = metadata["metrics"][best_name]
    tp = best_metrics["ConfusionMatrix"][1][1]
    fp = best_metrics["ConfusionMatrix"][0][1]
    fn = best_metrics["ConfusionMatrix"][1][0]
    
    # Revenue at risk
    gross_revenue_loss = total_churners * clv_annual
    
    # ML Campaign Costs & Savings
    targeted_customers = tp + fp
    total_campaign_cost = targeted_customers * offer_cost
    saved_customers = tp * conversion_rate
    gross_revenue_saved = saved_customers * clv_annual
    net_profit_saved = gross_revenue_saved - total_campaign_cost
    roi_percent = (net_profit_saved / total_campaign_cost) * 100 if total_campaign_cost > 0 else 0
    
    # Blanket Marketing Costs (Marketing to all 10k customers)
    blanket_campaign_cost = stats["total_records"] * offer_cost
    blanket_saved_customers = total_churners * conversion_rate
    blanket_gross_saved = blanket_saved_customers * clv_annual
    blanket_net_profit = blanket_gross_saved - blanket_campaign_cost
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Gross Revenue at Risk</div>
            <div class="metric-val" style="color: #ff1744;">€{gross_revenue_loss:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">ML Campaign Revenue Saved</div>
            <div class="metric-val" style="color: #00e676;">€{gross_revenue_saved:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Net Profit Saved (After Costs)</div>
            <div class="metric-val" style="color: #00b0ff;">€{net_profit_saved:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Campaign ROI</div>
            <div class="metric-val" style="color: {'#00e676' if roi_percent > 0 else '#ff1744'};">{roi_percent:+.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    st.subheader("Comparison: ML Precision Targeting vs. Blanket Marketing")
    
    comp_df = pd.DataFrame({
        "Campaign Strategy": ["ML Targeted Retention (Our Platform)", "Blanket Marketing (All Customers)", "No Retention Campaign"],
        "Targeted Customers": [f"{targeted_customers:,}", f"{stats['total_records']:,}", "0"],
        "Campaign Cost (€)": [f"€{total_campaign_cost:,.0f}", f"€{blanket_campaign_cost:,.0f}", "€0"],
        "Saved Customers": [f"{saved_customers:.0f}", f"{blanket_saved_customers:.0f}", "0"],
        "Net Value Created (€)": [f"€{net_profit_saved:,.0f}", f"€{blanket_net_profit:,.0f}", f"-€{gross_revenue_loss:,.0f}"]
    })
    st.table(comp_df)

# ==========================================
# MODULE 5: MODEL BENCHMARKS & THRESHOLDS
# ==========================================
elif module == "⚡ Model Benchmarks & Decision Thresholds":
    st.title("⚡ Model Benchmarks & Optimal Decision Thresholds")
    st.markdown("Compare candidate algorithms and adjust decision probability thresholds to balance Precision vs. Recall.")
    
    metrics_data = metadata["metrics"]
    numeric_metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "OptimalThreshold"]
    display_df = pd.DataFrame(metrics_data).T[numeric_metric_cols].astype(float)
    
    st.subheader("Candidate Model Evaluation Table")
    st.dataframe(display_df.style.highlight_max(axis=0, color='rgba(0, 230, 118, 0.35)'), use_container_width=True)
    
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("ROC Curves Comparison")
        fig_roc = go.Figure()
        for name, info in metadata.get("curves_data", {}).items():
            fig_roc.add_trace(go.Scatter(x=info["fpr"], y=info["tpr"], mode='lines', name=f"{name} (AUC={metrics_data[name]['ROC-AUC']:.3f})"))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash', color='gray'), name='Random Chance'))
        fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#ffffff')
        st.plotly_chart(fig_roc, use_container_width=True)
        
    with c2:
        st.subheader("Confusion Matrix (Best Model)")
        best_name = metadata["best_model_name"]
        cm = metrics_data[best_name]["ConfusionMatrix"]
        fig_cm = px.imshow(
            cm, text_auto=True,
            x=['Predicted Retained', 'Predicted Churned'],
            y=['Actual Retained', 'Actual Churned'],
            color_continuous_scale='Blues',
            title=f"Confusion Matrix for {best_name}"
        )
        fig_cm.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#ffffff')
        st.plotly_chart(fig_cm, use_container_width=True)

# ==========================================
# MODULE 6: SHAP EXPLAINABILITY
# ==========================================
elif module == "🔍 SHAP & Feature Explainability":
    st.title("🔍 Explainable AI & SHAP Driver Rankings")
    st.markdown("Audit-compliant explainability revealing key features driving customer churn risk.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Gini / Gain Feature Importances")
        importances = metadata.get("feature_importances", {})
        imp_df = pd.DataFrame(list(importances.items()), columns=['Feature', 'Importance']).head(12)
        fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Viridis')
        fig_imp.update_layout(yaxis=dict(autorange="reversed"), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#ffffff')
        st.plotly_chart(fig_imp, use_container_width=True)
        
    with col2:
        st.subheader("SHAP Global Impact (Mean |SHAP Value|)")
        shap_summary = metadata.get("shap_summary", {})
        shap_df = pd.DataFrame(list(shap_summary.items()), columns=['Feature', 'Mean_SHAP']).head(12)
        fig_shap = px.bar(shap_df, x='Mean_SHAP', y='Feature', orientation='h', color='Mean_SHAP', color_continuous_scale='Plasma')
        fig_shap.update_layout(yaxis=dict(autorange="reversed"), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#ffffff')
        st.plotly_chart(fig_shap, use_container_width=True)

# ==========================================
# MODULE 7: WHAT-IF SCENARIO SIMULATOR
# ==========================================
elif module == "🧪 What-If Scenario Simulator":
    st.title("🧪 Interactive What-If Scenario Simulator")
    st.markdown("Simulate retention interventions (e.g. activating a member, offering a 2nd product) and view immediate churn probability reduction.")
    
    st.subheader("Step 1: Define Baseline Customer Profile")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        sim_age = st.slider("Age", 18, 85, 48, key="s_age")
        sim_geo = st.selectbox("Geography", ["France", "Germany", "Spain"], index=1, key="s_geo")
        sim_salary = st.number_input("Salary (€)", value=90000.0, step=5000.0, key="s_salary")
    with sc2:
        sim_balance = st.number_input("Balance (€)", value=110000.0, step=5000.0, key="s_balance")
        sim_credit = st.slider("Credit Score", 300, 850, 610, key="s_credit")
        sim_tenure = st.slider("Tenure", 0, 10, 3, key="s_tenure")
    with sc3:
        sim_products = st.selectbox("Products", [1, 2, 3, 4], index=0, key="s_prod")
        sim_active = st.selectbox("Active Status", [0, 1], format_func=lambda x: "Inactive (0)" if x==0 else "Active (1)", key="s_act")
        sim_card = st.selectbox("Credit Card", [0, 1], index=1, key="s_card")

    base_inputs = {
        'CreditScore': sim_credit, 'Geography': sim_geo, 'Gender': 'Female',
        'Age': sim_age, 'Tenure': sim_tenure, 'Balance': sim_balance,
        'NumOfProducts': sim_products, 'HasCrCard': sim_card,
        'IsActiveMember': sim_active, 'EstimatedSalary': sim_salary
    }
    
    df_base, _ = preprocess_dataframe_batch(pd.DataFrame([base_inputs]))
    base_prob = best_model.predict_proba(df_base)[0][1]
    
    st.markdown("---")
    st.subheader("Step 2: Simulate Retention Offers")
    ic1, ic2 = st.columns(2)
    with ic1:
        new_active = st.radio("Simulate Activity Onboarding", [sim_active, 1 if sim_active==0 else 0], format_func=lambda x: "Keep Current" if x==sim_active else ("Activate Customer" if x==1 else "Deactivate"))
    with ic2:
        new_products = st.selectbox("Simulate Product Addition/Removal", [1, 2, 3, 4], index=sim_products-1)
        
    mod_inputs = base_inputs.copy()
    mod_inputs['IsActiveMember'] = new_active
    mod_inputs['NumOfProducts'] = new_products
    
    df_mod, _ = preprocess_dataframe_batch(pd.DataFrame([mod_inputs]))
    mod_prob = best_model.predict_proba(df_mod)[0][1]
    delta = (mod_prob - base_prob) * 100
    
    st.markdown("### 🎯 Simulation Results")
    res1, res2, res3 = st.columns(3)
    with res1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Baseline Churn Probability</div>
            <div class="metric-val" style="color: {'#ff1744' if base_prob > 0.5 else '#ffb300'};">{base_prob*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with res2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Post-Intervention Churn Risk</div>
            <div class="metric-val" style="color: {'#00e676' if mod_prob < 0.3 else '#ffb300'};">{mod_prob*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with res3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Risk Reduction Delta</div>
            <div class="metric-val" style="color: {'#00e676' if delta < 0 else '#ff1744'};">{delta:+.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
