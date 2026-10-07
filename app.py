import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Enterprise Page Configuration
st.set_page_config(
    page_title="Bank Customer Retention Portal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Corporate UI/UX Aesthetics
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
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
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
        background-color: rgba(0, 230, 118, 0.12);
        color: #00e676;
        border: 1px solid #00e676;
        padding: 6px 14px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }
    .risk-badge-medium {
        background-color: rgba(255, 179, 0, 0.12);
        color: #ffb300;
        border: 1px solid #ffb300;
        padding: 6px 14px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }
    .risk-badge-high {
        background-color: rgba(255, 23, 68, 0.12);
        color: #ff1744;
        border: 1px solid #ff1744;
        padding: 6px 14px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

ARTIFACT_DIR = "model_artifacts"
DATA_PATH = "European_Bank.csv"

@st.cache_resource
def load_models():
    metadata_path = os.path.join(ARTIFACT_DIR, "metadata.json")
    if not os.path.exists(metadata_path):
        st.error("System engine not ready. Run `python train_model.py` first.")
        st.stop()
        
    best_model = joblib.load(os.path.join(ARTIFACT_DIR, "best_churn_model.pkl"))
    scaler = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(ARTIFACT_DIR, "feature_names.pkl"))
    df_raw = pd.read_csv(DATA_PATH)
    
    return best_model, scaler, feature_names, df_raw

best_model, scaler, feature_names, df_raw = load_models()

def preprocess_input_df(df):
    """
    Transforms raw customer dataframe into the format expected by prediction engine.
    """
    df_clean = df.copy()
    
    cols_to_drop = [c for c in ['CustomerId', 'Surname', 'Year', 'Exited'] if c in df_clean.columns]
    df_feat = df_clean.drop(columns=cols_to_drop)
    
    df_feat = df_feat.fillna(df_feat.median(numeric_only=True))
    
    # Derived Feature Transformations
    df_feat['Balance_Salary_Ratio'] = df_feat['Balance'] / (df_feat['EstimatedSalary'] + 1.0)
    df_feat['Product_Density'] = df_feat['NumOfProducts'] / (df_feat['Tenure'] + 1.0)
    df_feat['Engagement_Product_Interaction'] = df_feat['IsActiveMember'] * df_feat['NumOfProducts']
    df_feat['Tenure_Age_Ratio'] = df_feat['Tenure'] / (df_feat['Age'] + 1.0)
    df_feat['Credit_Age_Ratio'] = df_feat['CreditScore'] / (df_feat['Age'] + 1.0)
    df_feat['Is_Zero_Balance'] = (df_feat['Balance'] == 0).astype(int)
    df_feat['High_Risk_Age_Group'] = ((df_feat['Age'] >= 38) & (df_feat['Age'] <= 60)).astype(int)
    
    if 'Geography' in df_feat.columns:
        df_feat['Geography_France'] = (df_feat['Geography'] == 'France').astype(int)
        df_feat['Geography_Germany'] = (df_feat['Geography'] == 'Germany').astype(int)
        df_feat['Geography_Spain'] = (df_feat['Geography'] == 'Spain').astype(int)
        df_feat = df_feat.drop(columns=['Geography'])
        
    if 'Gender' in df_feat.columns:
        df_feat['Gender_Female'] = (df_feat['Gender'] == 'Female').astype(int)
        df_feat['Gender_Male'] = (df_feat['Gender'] == 'Male').astype(int)
        df_feat = df_feat.drop(columns=['Gender'])
        
    for col in feature_names:
        if col not in df_feat.columns:
            df_feat[col] = 0
            
    df_feat = df_feat[feature_names]
    return df_feat

def get_action_recommendation(row):
    if row['IsActiveMember'] == 0:
        return "Digital App Bonus (0.5% Deposit Bonus on App Login)"
    elif row['NumOfProducts'] == 1:
        return "2nd Product Promotion (Free Credit Card / Savings Add-on)"
    elif row['NumOfProducts'] >= 3:
        return "Relationship Manager Call (Consolidate Accounts & Reduce Fees)"
    elif row['Geography_Germany'] == 1:
        return "Germany Regional VIP Loyalty Program"
    elif row['High_Risk_Age_Group'] == 1:
        return "Personalized Wealth & Mortgage Refinancing Consultation"
    elif row['Is_Zero_Balance'] == 1:
        return "Direct Deposit Salary Bonus ($50 Welcome Deposit)"
    else:
        return "Standard Promotional Engagement"

# Corporate Sidebar Navigation
st.sidebar.title("Bank Retention Portal")
st.sidebar.markdown("Customer Retention & Risk Management Engine")

user_tool = st.sidebar.radio(
    "Select Portal Module",
    [
        "Batch Risk Scoring & CSV Export",
        "Individual Risk Evaluation",
        "Scenario Simulator",
        "Financial ROI Calculator"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Retail Banking Risk Governance")

# ==========================================
# MODULE 1: BATCH RISK SCORING & EXPORT
# ==========================================
if user_tool == "Batch Risk Scoring & CSV Export":
    st.title("Batch Customer Risk Scoring")
    st.markdown("Upload a customer file (CSV) to compute risk scores, map retention strategies, and export actionable CSV lists for campaign management.")
    
    col_up, col_sample = st.columns([3, 1])
    with col_up:
        uploaded_file = st.file_uploader("Select Customer Dataset (CSV Format)", type=["csv"])
    with col_sample:
        sample_csv = df_raw.head(100).drop(columns=['Exited']).to_csv(index=False).encode('utf-8')
        st.write("")
        st.write("")
        st.download_button("Download Sample CSV Template", data=sample_csv, file_name="sample_customers.csv", mime="text/csv")
        
    if uploaded_file is not None:
        df_user = pd.read_csv(uploaded_file)
        st.success(f"Successfully loaded {len(df_user):,} customer records.")
        
        with st.spinner("Processing batch risk evaluation..."):
            df_feat = preprocess_input_df(df_user)
            probs = best_model.predict_proba(df_feat)[:, 1]
            
            df_scored = df_user.copy()
            df_scored['Risk_Score_%'] = np.round(probs * 100, 1)
            df_scored['Risk_Level'] = pd.cut(
                probs, bins=[-0.01, 0.30, 0.60, 1.01],
                labels=['Low Risk', 'Medium Risk', 'High Risk']
            )
            df_scored['Recommended_Strategy'] = df_feat.apply(get_action_recommendation, axis=1)
            
        st.markdown("---")
        st.subheader("Portfolio Risk Distribution")
        
        m1, m2, m3, m4 = st.columns(4)
        total = len(df_scored)
        high = (df_scored['Risk_Level'] == 'High Risk').sum()
        med = (df_scored['Risk_Level'] == 'Medium Risk').sum()
        low = (df_scored['Risk_Level'] == 'Low Risk').sum()
        
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Customers</div>
                <div class="metric-val" style="color: #00b0ff;">{total:,}</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">High Risk (>60%)</div>
                <div class="metric-val" style="color: #ff1744;">{high:,}</div>
                <div style="color: #ff1744;">{high/total*100:.1f}% of portfolio</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Medium Risk (30-60%)</div>
                <div class="metric-val" style="color: #ffb300;">{med:,}</div>
                <div style="color: #ffb300;">{med/total*100:.1f}% of portfolio</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Low Risk (<30%)</div>
                <div class="metric-val" style="color: #00e676;">{low:,}</div>
                <div style="color: #00e676;">{low/total*100:.1f}% of portfolio</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        st.subheader("Filter & Export Scored Customer List")
        
        selected_tiers = st.multiselect(
            "Filter Table by Risk Tier:",
            ['High Risk', 'Medium Risk', 'Low Risk'],
            default=['High Risk', 'Medium Risk']
        )
        
        display_df = df_scored[df_scored['Risk_Level'].isin(selected_tiers)]
        st.dataframe(display_df, use_container_width=True)
        
        # CSV Export
        export_csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Export Scored Customer Action Plan (CSV)",
            data=export_csv,
            file_name="retention_action_plan.csv",
            mime="text/csv"
        )
    else:
        st.info("Select a customer CSV file above or download the sample template to evaluate batch risk.")

# ==========================================
# MODULE 2: INDIVIDUAL RISK EVALUATION
# ==========================================
elif user_tool == "Individual Risk Evaluation":
    st.title("Individual Risk Evaluation")
    st.markdown("Specify customer parameters to compute the risk score and view targeted retention strategies.")
    
    with st.form("single_cust_form"):
        st.subheader("Customer Parameters")
        c1, c2, c3 = st.columns(3)
        with c1:
            credit_score = st.slider("Credit Score", 300, 850, 640)
            geography = st.selectbox("Country", ["France", "Germany", "Spain"])
            gender = st.selectbox("Gender", ["Female", "Male"])
            age = st.slider("Age", 18, 90, 44)
        with c2:
            tenure = st.slider("Tenure (Years with Bank)", 0, 10, 3)
            balance = st.number_input("Account Balance (€)", min_value=0.0, max_value=300000.0, value=85000.0, step=5000.0)
            salary = st.number_input("Estimated Annual Salary (€)", min_value=0.0, max_value=250000.0, value=90000.0, step=5000.0)
        with c3:
            num_products = st.selectbox("Number of Products", [1, 2, 3, 4], index=0)
            has_card = st.selectbox("Has Credit Card", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")
            is_active = st.selectbox("Member Activity Status", [1, 0], format_func=lambda x: "Active (1)" if x==1 else "Inactive (0)")
            
        submit_btn = st.form_submit_button("Evaluate Risk Score")
        
    cust_inputs = {
        'CreditScore': credit_score, 'Geography': geography, 'Gender': gender,
        'Age': age, 'Tenure': tenure, 'Balance': balance,
        'NumOfProducts': num_products, 'HasCrCard': has_card,
        'IsActiveMember': is_active, 'EstimatedSalary': salary
    }
    
    df_single = pd.DataFrame([cust_inputs])
    df_feat = preprocess_input_df(df_single)
    prob = best_model.predict_proba(df_feat)[0][1]
    
    st.markdown("---")
    r1, r2 = st.columns([1, 2])
    with r1:
        color = "#ff1744" if prob > 0.6 else ("#ffb300" if prob >= 0.3 else "#00e676")
        tier = "High Risk" if prob > 0.6 else ("Medium Risk" if prob >= 0.3 else "Low Risk")
        badge = "risk-badge-high" if prob > 0.6 else ("risk-badge-medium" if prob >= 0.3 else "risk-badge-low")
        st.markdown(f"""
        <div class="metric-card" style="padding: 30px;">
            <div class="metric-label">Calculated Risk Score</div>
            <div class="metric-val" style="font-size: 3.4rem; color: {color};">{prob*100:.1f}%</div>
            <div class="{badge}">{tier}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with r2:
        action = get_action_recommendation(df_feat.iloc[0])
        st.markdown("### Recommended Retention Strategy")
        st.info(f"**Action**: {action}")
        
        st.markdown("#### Primary Risk Factors:")
        recs = []
        if is_active == 0:
            recs.append("- **Member Inactivity**: Customer exhibits low digital/branch engagement.")
        if num_products == 1:
            recs.append("- **Single Product Ownership**: Vulnerable to competitor conversion offers.")
        elif num_products >= 3:
            recs.append("- **Product Overcrowding**: Customer holds 3-4 products (associated with high fee dissatisfaction).")
        if geography == "Germany":
            recs.append("- **Germany Demographic**: German accounts have higher baseline churn propensity.")
        if age >= 38 and age <= 60:
            recs.append("- **Age Bracket**: Customer is within the 38-60 age window with elevated flight risk.")
        if balance == 0:
            recs.append("- **Zero Balance**: Customer maintains no active account balance.")
            
        if recs:
            for r in recs:
                st.markdown(r)
        else:
            st.success("Customer profile indicates strong retention probability.")

# ==========================================
# MODULE 3: SCENARIO SIMULATOR
# ==========================================
elif user_tool == "Scenario Simulator":
    st.title("Scenario Simulator")
    st.markdown("Evaluate how specific promotional offers or activity activation impact customer risk scores in real time.")
    
    st.subheader("1. Baseline Profile")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        sim_age = st.slider("Age", 18, 85, 46, key="wf_age")
        sim_geo = st.selectbox("Country", ["France", "Germany", "Spain"], index=1, key="wf_geo")
        sim_salary = st.number_input("Salary (€)", value=85000.0, step=5000.0, key="wf_salary")
    with sc2:
        sim_balance = st.number_input("Balance (€)", value=100000.0, step=5000.0, key="wf_balance")
        sim_credit = st.slider("Credit Score", 300, 850, 620, key="wf_credit")
        sim_tenure = st.slider("Tenure (Years)", 0, 10, 4, key="wf_tenure")
    with sc3:
        sim_products = st.selectbox("Current Products", [1, 2, 3, 4], index=0, key="wf_prod")
        sim_active = st.selectbox("Current Activity", [0, 1], format_func=lambda x: "Inactive (0)" if x==0 else "Active (1)", key="wf_act")
        sim_card = st.selectbox("Credit Card", [0, 1], index=1, key="wf_card")

    base_dict = {
        'CreditScore': sim_credit, 'Geography': sim_geo, 'Gender': 'Female',
        'Age': sim_age, 'Tenure': sim_tenure, 'Balance': sim_balance,
        'NumOfProducts': sim_products, 'HasCrCard': sim_card,
        'IsActiveMember': sim_active, 'EstimatedSalary': sim_salary
    }
    
    df_base = preprocess_input_df(pd.DataFrame([base_dict]))
    base_prob = best_model.predict_proba(df_base)[0][1]
    
    st.markdown("---")
    st.subheader("2. Simulated Interventions")
    ic1, ic2 = st.columns(2)
    with ic1:
        new_active = st.radio(
            "Simulate Member Activity:",
            [sim_active, 1 if sim_active==0 else 0],
            format_func=lambda x: "Keep Current" if x==sim_active else ("Activate Member" if x==1 else "Deactivate Member")
        )
    with ic2:
        new_products = st.selectbox("Simulate Product Count Adjustment:", [1, 2, 3, 4], index=sim_products-1)
        
    mod_dict = base_dict.copy()
    mod_dict['IsActiveMember'] = new_active
    mod_dict['NumOfProducts'] = new_products
    
    df_mod = preprocess_input_df(pd.DataFrame([mod_dict]))
    mod_prob = best_model.predict_proba(df_mod)[0][1]
    delta = (mod_prob - base_prob) * 100
    
    st.markdown("### Simulation Impact Analysis")
    res1, res2, res3 = st.columns(3)
    with res1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Baseline Churn Risk</div>
            <div class="metric-val" style="color: {'#ff1744' if base_prob > 0.5 else '#ffb300'};">{base_prob*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with res2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Post-Intervention Risk</div>
            <div class="metric-val" style="color: {'#00e676' if mod_prob < 0.3 else '#ffb300'};">{mod_prob*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with res3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Risk Reduction Delta</div>
            <div class="metric-val" style="color: {'#00e676' if delta < 0 else '#ff1744'};">{delta:+.1f}%</div>
            <div>{"Risk Reduced" if delta < 0 else "Risk Increased"}</div>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# MODULE 4: FINANCIAL ROI CALCULATOR
# ==========================================
elif user_tool == "Financial ROI Calculator":
    st.title("Financial ROI Calculator")
    st.markdown("Quantify financial returns and saved revenue from targeted retention campaigns.")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        annual_val = st.number_input("Average Annual Customer Revenue (€)", value=1200.0, step=100.0)
    with c2:
        offer_cost = st.number_input("Retention Offer Cost per Customer (€)", value=150.0, step=25.0)
    with c3:
        conv_rate = st.slider("Campaign Acceptance Rate (%)", 10, 80, 40) / 100.0
        
    st.markdown("---")
    
    total_customers = len(df_raw)
    total_churners = int(df_raw['Exited'].sum())
    
    df_all_feat = preprocess_input_df(df_raw.drop(columns=['Exited']))
    all_probs = best_model.predict_proba(df_all_feat)[:, 1]
    
    predicted_high_risk = (all_probs >= 0.5).sum()
    actual_churners_caught = ((all_probs >= 0.5) & (df_raw['Exited'] == 1)).sum()
    
    revenue_at_risk = total_churners * annual_val
    campaign_cost = predicted_high_risk * offer_cost
    saved_customers = actual_churners_caught * conv_rate
    revenue_saved = saved_customers * annual_val
    net_profit = revenue_saved - campaign_cost
    roi = (net_profit / campaign_cost * 100) if campaign_cost > 0 else 0
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Revenue at Risk</div>
            <div class="metric-val" style="color: #ff1744;">€{revenue_at_risk:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Targeted At-Risk Customers</div>
            <div class="metric-val" style="color: #00b0ff;">{predicted_high_risk:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Net Saved Revenue (Post-Costs)</div>
            <div class="metric-val" style="color: #00e676;">€{net_profit:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Campaign ROI</div>
            <div class="metric-val" style="color: {'#00e676' if roi > 0 else '#ff1744'};">{roi:+.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
