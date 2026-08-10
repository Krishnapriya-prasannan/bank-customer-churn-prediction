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
    page_title="Bank Customer Churn Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design & Glassmorphism Aesthetics
st.markdown("""
<style>
    /* Dark Theme Custom Colors */
    :root {
        --bg-main: #0e1117;
        --card-bg: #1a1f2c;
        --card-border: #2a3142;
        --text-primary: #ffffff;
        --text-secondary: #90a4ae;
        --accent-blue: #00b0ff;
        --accent-green: #00e676;
        --accent-yellow: #ffb300;
        --accent-red: #ff1744;
    }
    
    /* Main App Background */
    .stApp {
        background-color: var(--bg-main);
        color: var(--text-primary);
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(26,31,44,0.9), rgba(35,43,62,0.8));
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
        font-size: 0.9rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    
    /* Custom Risk Badges */
    .risk-badge-low {
        background-color: rgba(0, 230, 118, 0.15);
        color: #00e676;
        border: 1px solid #00e676;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .risk-badge-medium {
        background-color: rgba(255, 179, 0, 0.15);
        color: #ffb300;
        border: 1px solid #ffb300;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .risk-badge-high {
        background-color: rgba(255, 23, 68, 0.15);
        color: #ff1744;
        border: 1px solid #ff1744;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    
    /* Sidebar Styling */
    .css-1d35500, [data-testid="stSidebar"] {
        background-color: #131722;
        border-right: 1px solid #232838;
    }
    
    /* Section Containers */
    .section-box {
        background: #1a1f2c;
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #283044;
        margin-bottom: 20px;
    }
    
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

ARTIFACT_DIR = "model_artifacts"
DATA_PATH = "European_Bank.csv"

@st.cache_resource
def load_artifacts():
    metadata_path = os.path.join(ARTIFACT_DIR, "metadata.json")
    if not os.path.exists(metadata_path):
        st.error("Model artifacts not found! Please run `python train_model.py` first.")
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

# Helper Function: Create Feature Vector from raw inputs
def prepare_customer_feature_vector(inputs):
    # Base dictionary
    data = {
        'CreditScore': inputs['CreditScore'],
        'Age': inputs['Age'],
        'Tenure': inputs['Tenure'],
        'Balance': inputs['Balance'],
        'NumOfProducts': inputs['NumOfProducts'],
        'HasCrCard': inputs['HasCrCard'],
        'IsActiveMember': inputs['IsActiveMember'],
        'EstimatedSalary': inputs['EstimatedSalary'],
        
        # Engineered Features
        'Balance_Salary_Ratio': inputs['Balance'] / (inputs['EstimatedSalary'] + 1.0),
        'Product_Density': inputs['NumOfProducts'] / (inputs['Tenure'] + 1.0),
        'Engagement_Product_Interaction': inputs['IsActiveMember'] * inputs['NumOfProducts'],
        'Tenure_Age_Ratio': inputs['Tenure'] / (inputs['Age'] + 1.0),
        'Credit_Age_Ratio': inputs['CreditScore'] / (inputs['Age'] + 1.0),
        'Is_Zero_Balance': 1 if inputs['Balance'] == 0 else 0,
        'High_Risk_Age_Group': 1 if (inputs['Age'] >= 38 and inputs['Age'] <= 60) else 0,
        
        # One-Hot Encoding
        'Geography_France': 1 if inputs['Geography'] == 'France' else 0,
        'Geography_Germany': 1 if inputs['Geography'] == 'Germany' else 0,
        'Geography_Spain': 1 if inputs['Geography'] == 'Spain' else 0,
        'Gender_Female': 1 if inputs['Gender'] == 'Female' else 0,
        'Gender_Male': 1 if inputs['Gender'] == 'Male' else 0,
    }
    
    df_vec = pd.DataFrame([data])
    # Ensure exact column ordering as trained model
    df_vec = df_vec[feature_names]
    return df_vec

# Calculate prediction & risk drivers
def predict_churn_risk(df_vec):
    prob = best_model.predict_proba(df_vec)[0][1]
    
    if prob < 0.30:
        level = "Low Risk"
        badge_class = "risk-badge-low"
    elif prob <= 0.60:
        level = "Medium Risk"
        badge_class = "risk-badge-medium"
    else:
        level = "High Risk"
        badge_class = "risk-badge-high"
        
    return prob, level, badge_class

# Sidebar Navigation
st.sidebar.title("🏦 Churn Intelligence")
st.sidebar.markdown("European Retail Banking Predictive Analytics")

module = st.sidebar.radio(
    "Navigation Modules",
    [
        "📊 Executive Overview",
        "🧮 Churn Risk Calculator",
        "⚡ Model Performance Benchmark",
        "🔍 SHAP & Feature Explainability",
        "🧪 What-If Scenario Simulator"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption(f"**Selected Model:** {metadata['best_model_name']}")
st.sidebar.caption(f"**ROC-AUC Score:** {metadata['metrics'][metadata['best_model_name']]['ROC-AUC']:.4f}")
st.sidebar.caption("© European Central Bank Churn Analytics System")

# ==========================================
# MODULE 1: EXECUTIVE OVERVIEW
# ==========================================
if module == "📊 Executive Overview":
    st.title("📊 Retail Bank Customer Churn Overview")
    st.markdown("Macro-level statistics, customer demographics, and financial retention metrics across European markets.")
    
    # Top KPI Metrics
    stats = metadata["dataset_stats"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Bank Customers</div>
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
            <div class="metric-label">Retained Customers</div>
            <div class="metric-val" style="color: #00e676;">{stats['retained_records']:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Churned Customers</div>
            <div class="metric-val" style="color: #ffb300;">{stats['churn_records']:,}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # Visual Analytics Charts
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Geography & Demographic Churn Breakdown")
        geo_df = df_raw.groupby(['Geography', 'Exited']).size().reset_index(name='Count')
        geo_df['Status'] = geo_df['Exited'].map({0: 'Retained', 1: 'Churned'})
        
        fig_geo = px.bar(
            geo_df, x='Geography', y='Count', color='Status', barmode='group',
            color_discrete_map={'Retained': '#00e676', 'Churned': '#ff1744'},
            title="Customer Churn Count by Country"
        )
        fig_geo.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#ffffff')
        st.plotly_chart(fig_geo, use_container_width=True)
        
    with col_right:
        st.subheader("Product Utilization vs. Churn Rate")
        prod_df = df_raw.groupby('NumOfProducts')['Exited'].agg(['count', 'mean']).reset_index()
        prod_df['ChurnRate%'] = prod_df['mean'] * 100
        
        fig_prod = px.bar(
            prod_df, x='NumOfProducts', y='ChurnRate%', text='ChurnRate%',
            color='ChurnRate%', color_continuous_scale='Reds',
            title="Churn Probability by Number of Bank Products Owned"
        )
        fig_prod.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_prod.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#ffffff')
        st.plotly_chart(fig_prod, use_container_width=True)

    # Age Distribution by Churn Status
    st.subheader("Age Distribution & Churn Vulnerability Window")
    fig_age = px.histogram(
        df_raw, x='Age', color='Exited', marginal='box', nbins=30,
        color_discrete_map={0: '#00e676', 1: '#ff1744'},
        title="Customer Age Distribution (Green: Retained, Red: Churned)"
    )
    fig_age.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#ffffff')
    st.plotly_chart(fig_age, use_container_width=True)

# ==========================================
# MODULE 2: CHURN RISK CALCULATOR
# ==========================================
elif module == "🧮 Churn Risk Calculator":
    st.title("🧮 Individual Customer Churn Risk Calculator")
    st.markdown("Input customer demographic, account, and product details to generate real-time churn probabilities and custom retention action plans.")
    
    with st.form("customer_risk_form"):
        st.subheader("Customer Profile & Account Inputs")
        c1, c2, c3 = st.columns(3)
        
        with c1:
            credit_score = st.slider("Credit Score", 300, 850, 650)
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
            gender = st.selectbox("Gender", ["Female", "Male"])
            age = st.slider("Customer Age", 18, 92, 38)
            
        with c2:
            tenure = st.slider("Tenure (Years with Bank)", 0, 10, 5)
            balance = st.number_input("Account Balance (€)", min_value=0.0, max_value=300000.0, value=75000.0, step=5000.0)
            salary = st.number_input("Estimated Annual Salary (€)", min_value=0.0, max_value=250000.0, value=100000.0, step=5000.0)
            
        with c3:
            num_products = st.selectbox("Number of Products", [1, 2, 3, 4], index=0)
            has_crcard = st.selectbox("Has Credit Card?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
            is_active = st.selectbox("Is Active Member?", [1, 0], format_func=lambda x: "Yes (Active)" if x == 1 else "No (Inactive)")
            
        submit_calc = st.form_submit_button("⚡ Calculate Churn Probability")
        
    if submit_calc or True:  # Default display
        inputs = {
            'CreditScore': credit_score, 'Geography': geography, 'Gender': gender,
            'Age': age, 'Tenure': tenure, 'Balance': balance,
            'NumOfProducts': num_products, 'HasCrCard': has_crcard,
            'IsActiveMember': is_active, 'EstimatedSalary': salary
        }
        
        df_vec = prepare_customer_feature_vector(inputs)
        prob, level, badge_class = predict_churn_risk(df_vec)
        
        st.markdown("---")
        st.subheader("Risk Score Output & Retention Recommendations")
        
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.markdown(f"""
            <div class="metric-card" style="padding: 30px;">
                <div class="metric-label">Estimated Churn Risk</div>
                <div class="metric-val" style="font-size: 3.2rem; color: {'#ff1744' if prob > 0.6 else ('#ffb300' if prob >= 0.3 else '#00e676')};">
                    {prob*100:.1f}%
                </div>
                <div class="{badge_class}">{level}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with res_col2:
            st.markdown("#### Key Churn Drivers & Automated Recommendations")
            
            recs = []
            if is_active == 0:
                recs.append("⚠️ **Inactive Member Status**: Customer has low bank interaction. **Action**: Offer a 0.5% interest bonus on deposits upon logging into mobile banking 3x this month.")
            if num_products == 1:
                recs.append("⚠️ **Single Product Vulnerability**: Customer has only 1 product. **Action**: Target with pre-approved credit card or investment account onboarding.")
            elif num_products >= 3:
                recs.append("🔴 **Product Overcrowding**: Customer holds 3-4 products (historically highest churn probability). **Action**: Initiate direct Relationship Manager call to consolidate accounts.")
            if geography == "Germany":
                recs.append("🌐 **Regional Risk Window**: German account holders exhibit higher churn rate. **Action**: Enroll in Germany VIP loyalty tier.")
            if age >= 38 and age <= 60:
                recs.append("👤 **Prime Churn Demographic**: Customer is in the 38–60 age group. **Action**: Offer tailored wealth retention & mortgage refinancing plans.")
            if balance == 0:
                recs.append("💵 **Zero Account Balance**: High flight risk. **Action**: Send automated salary direct deposit incentive campaign.")
                
            if not recs:
                st.success("✅ Excellent customer retention profile. Maintain standard promotional engagement.")
            else:
                for r in recs:
                    st.markdown(f"- {r}")

# ==========================================
# MODULE 3: MODEL PERFORMANCE BENCHMARK
# ==========================================
elif module == "⚡ Model Performance Benchmark":
    st.title("⚡ Machine Learning Model Comparison & Metrics")
    st.markdown("Empirical benchmark results evaluating Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost.")
    
    # Metrics Table
    metrics_data = metadata["metrics"]
    metrics_df = pd.DataFrame(metrics_data).T
    
    # Exclude non-numeric columns like ConfusionMatrix for table styling
    numeric_metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    display_df = metrics_df[numeric_metric_cols].astype(float)
    
    st.subheader("Model Evaluation Summary Table")
    st.dataframe(
        display_df.style.highlight_max(axis=0, color='rgba(0, 230, 118, 0.35)'),
        use_container_width=True
    )
    
    st.markdown("---")
    
    c_roc, c_cm = st.columns(2)
    
    with c_roc:
        st.subheader("Receiver Operating Characteristic (ROC) Curves")
        fig_roc = go.Figure()
        
        curves_data = metadata.get("curves_data", {})
        for model_name, curve_info in curves_data.items():
            fig_roc.add_trace(go.Scatter(
                x=curve_info["fpr"], y=curve_info["tpr"],
                mode='lines', name=f"{model_name} (AUC = {metrics_data[model_name]['ROC-AUC']:.3f})"
            ))
            
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'), name='Random Chance'))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#ffffff'
        )
        st.plotly_chart(fig_roc, use_container_width=True)
        
    with c_cm:
        st.subheader("Confusion Matrix (Best Model)")
        best_name = metadata["best_model_name"]
        cm_matrix = metrics_data[best_name]["ConfusionMatrix"]
        
        fig_cm = px.imshow(
            cm_matrix, text_auto=True,
            x=['Predicted Retained', 'Predicted Churned'],
            y=['Actual Retained', 'Actual Churned'],
            color_continuous_scale='Blues',
            title=f"Confusion Matrix for {best_name}"
        )
        fig_cm.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#ffffff')
        st.plotly_chart(fig_cm, use_container_width=True)

# ==========================================
# MODULE 4: SHAP & EXPLAINABILITY
# ==========================================
elif module == "🔍 SHAP & Feature Explainability":
    st.title("🔍 Machine Learning Model Interpretability")
    st.markdown("Explainable AI (XAI) insights revealing key risk drivers behind customer churn predictions.")
    
    col_feat, col_shap = st.columns(2)
    
    with col_feat:
        st.subheader("Gini / Gain Feature Importance Ranking")
        importances = metadata.get("feature_importances", {})
        imp_df = pd.DataFrame(list(importances.items()), columns=['Feature', 'Importance']).head(12)
        
        fig_imp = px.bar(
            imp_df, x='Importance', y='Feature', orientation='h',
            color='Importance', color_continuous_scale='Viridis',
            title="Top Feature Importance Scores"
        )
        fig_imp.update_layout(yaxis=dict(autorange="reversed"), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#ffffff')
        st.plotly_chart(fig_imp, use_container_width=True)
        
    with col_shap:
        st.subheader("SHAP Global Impact (Mean |SHAP Value|)")
        shap_summary = metadata.get("shap_summary", {})
        shap_df = pd.DataFrame(list(shap_summary.items()), columns=['Feature', 'Mean_SHAP']).head(12)
        
        fig_shap = px.bar(
            shap_df, x='Mean_SHAP', y='Feature', orientation='h',
            color='Mean_SHAP', color_continuous_scale='Plasma',
            title="Feature Impact on Model Output Magnitude"
        )
        fig_shap.update_layout(yaxis=dict(autorange="reversed"), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#ffffff')
        st.plotly_chart(fig_shap, use_container_width=True)

    st.markdown("---")
    st.subheader("Key Business Takeaways from SHAP Analysis")
    st.markdown("""
    1. **Age Vulnerability**: Customer Age and `High_Risk_Age_Group` are the single strongest predictors of churn. Customers between 38 and 60 years old churn at significantly higher rates.
    2. **Product Over-saturation**: Having 3 or 4 products dramatically increases churn probability, while 2 products represents the optimal stickiness point.
    3. **Active Membership**: Active bank engagement cuts churn probability in half across all customer demographics.
    4. **Geographic Variance**: German customers exhibit higher baseline churn propensity compared to French and Spanish counterparts.
    """)

# ==========================================
# MODULE 5: WHAT-IF SCENARIO SIMULATOR
# ==========================================
elif module == "🧪 What-If Scenario Simulator":
    st.title("🧪 Interactive What-If Scenario Simulator")
    st.markdown("Simulate proactive retention interventions (e.g., activating a member, offering cross-sell products, boosting balance) and observe immediate churn risk reduction.")
    
    st.subheader("Step 1: Define Baseline Customer Profile")
    sc1, sc2, sc3 = st.columns(3)
    
    with sc1:
        sim_age = st.slider("Age", 18, 85, 48, key="sim_age")
        sim_geo = st.selectbox("Geography", ["France", "Germany", "Spain"], index=1, key="sim_geo")
        sim_salary = st.number_input("Salary (€)", value=90000.0, step=5000.0, key="sim_salary")
        
    with sc2:
        sim_balance = st.number_input("Balance (€)", value=110000.0, step=5000.0, key="sim_balance")
        sim_credit = st.slider("Credit Score", 300, 850, 610, key="sim_credit")
        sim_tenure = st.slider("Tenure", 0, 10, 3, key="sim_tenure")
        
    with sc3:
        sim_products = st.selectbox("Num Of Products", [1, 2, 3, 4], index=0, key="sim_prod")
        sim_active = st.selectbox("Active Member Status", [0, 1], format_func=lambda x: "Inactive (0)" if x==0 else "Active (1)", key="sim_act")
        sim_card = st.selectbox("Has Credit Card", [0, 1], index=1, key="sim_card")

    baseline_inputs = {
        'CreditScore': sim_credit, 'Geography': sim_geo, 'Gender': 'Female',
        'Age': sim_age, 'Tenure': sim_tenure, 'Balance': sim_balance,
        'NumOfProducts': sim_products, 'HasCrCard': sim_card,
        'IsActiveMember': sim_active, 'EstimatedSalary': sim_salary
    }
    
    df_base = prepare_customer_feature_vector(baseline_inputs)
    base_prob, base_level, _ = predict_churn_risk(df_base)
    
    st.markdown("---")
    st.subheader("Step 2: Simulate Retention Interventions")
    
    int_c1, int_c2 = st.columns(2)
    
    with int_c1:
        new_active = st.radio("Simulate Activity Onboarding", [sim_active, 1 if sim_active==0 else 0],
                              format_func=lambda x: "Keep Current" if x==sim_active else ("Activate Customer" if x==1 else "Deactivate Customer"))
        
    with int_c2:
        new_products = st.selectbox("Simulate Product Addition/Removal", [1, 2, 3, 4], index=sim_products-1)
        
    modified_inputs = baseline_inputs.copy()
    modified_inputs['IsActiveMember'] = new_active
    modified_inputs['NumOfProducts'] = new_products
    
    df_mod = prepare_customer_feature_vector(modified_inputs)
    mod_prob, mod_level, _ = predict_churn_risk(df_mod)
    
    delta = (mod_prob - base_prob) * 100
    
    st.markdown("### 🎯 Simulation Results")
    res_c1, res_c2, res_c3 = st.columns(3)
    
    with res_c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Baseline Churn Probability</div>
            <div class="metric-val" style="color: #ff1744 if base_prob > 0.5 else '#ffb300';">{base_prob*100:.1f}%</div>
            <div>Status: {base_level}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with res_c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Post-Intervention Churn Risk</div>
            <div class="metric-val" style="color: {'#00e676' if mod_prob < 0.3 else '#ffb300'};">{mod_prob*100:.1f}%</div>
            <div>Status: {mod_level}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with res_c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Risk Reduction Delta</div>
            <div class="metric-val" style="color: {'#00e676' if delta < 0 else '#ff1744'};">{delta:+.1f}%</div>
            <div>{"🎉 Risk Reduced" if delta < 0 else "⚠️ Risk Increased"}</div>
        </div>
        """, unsafe_allow_html=True)
