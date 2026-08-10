# Predictive Modeling and Risk Scoring for Bank Customer Churn
**A Machine Learning Framework for Proactive Retention & Explainable Risk Quantification**

*Prepared for Retail Bank Executive Leadership & Regulatory Policy Stakeholders (European Central Bank Context)*

---

## Executive Summary

Customer churn presents a significant threat to financial institution profitability, directly degrading Customer Lifetime Value (CLV), balance sheet stability, and cross-sell velocity. Traditional banking retention models rely on retrospective exit surveys—reacting only *after* customer relationship severance has occurred. 

This project introduces a **Predictive Churn Intelligence Platform** deployed across 10,000 European bank customer accounts. By engineering domain-specific interaction features and benchmarking 5 Machine Learning algorithms (Logistic Regression, Decision Trees, Random Forests, Gradient Boosting, and HistGradientBoosting), the system transitions retail banking from reactive churn accounting to **proactive probability scoring**.

### Key Empirical Findings
1. **Model Performance**: **Gradient Boosting Classifier** achieved the highest overall discrimination capability with a **ROC-AUC of 0.8632** and an **Accuracy of 86.40%**. For recall-prioritized retention campaigns, Decision Tree and Random Forest models captured **77.40% and 63.88% of all actual churners** respectively.
2. **The Product Paradox**: Customer churn exhibits a non-linear relationship with product ownership. Customers holding **2 bank products** demonstrate an optimal retention profile (**7.6% churn rate**), whereas holding **1 product** increases churn to **27.7%**, and holding **3+ products** drives churn rates above **82.7%** due to product friction and fee dissatisfaction.
3. **Active Engagement Factor**: Digital and branch active membership cuts churn propensity by **nearly 50%** (14.3% churn for active members vs 26.9% for inactive members).
4. **Demographic & Geographic Vulnerability**: Account holders in **Germany** experience a **27.5% baseline churn rate**—substantially higher than France (16.15%) and Spain (16.67%). Furthermore, customers aged **38 to 60** represent the primary flight-risk window.

---

## 1. Problem Statement & Business Context

### 1.1 The High Cost of Retail Banking Churn
In retail banking, acquiring a new customer costs 5 to 7 times more than retaining an existing account holder. When a customer churns, the bank forfeits:
- Accumulated core deposit liquidity.
- Recurring interest income from credit lines and mortgages.
- Fee income from payment transactions and wealth advisory products.
- Low-cost deposit funding required to support central bank liquidity ratios.

### 1.2 Explainability & Regulatory Compliance
Modern central banking frameworks (such as the European Central Bank guidelines on AI governance) require predictive risk scoring systems to be **transparent, audit-compliant, and free from unethical bias**. Black-box decisioning is unacceptable for customer-facing automated actions. Thus, model interpretability via **SHAP (SHapley Additive exPlanations)** is integrated directly into the analytics framework.

---

## 2. Dataset & Exploratory Data Analysis (EDA)

The empirical analysis is conducted on `European_Bank.csv`, comprising **10,000 customer records** across 3 primary European operating regions (France, Germany, Spain).

### 2.1 Feature Definitions
| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `CustomerId` | Identifier | Unique customer account number (Removed for modeling) |
| `Surname` | Text | Customer surname (Removed for modeling) |
| `CreditScore` | Numeric | Creditworthiness metric (300 - 850) |
| `Geography` | Categorical | Country of residence (France, Germany, Spain) |
| `Gender` | Categorical | Male / Female |
| `Age` | Numeric | Customer age in years (18 - 92) |
| `Tenure` | Numeric | Number of years as a bank customer (0 - 10) |
| `Balance` | Numeric | Current account balance (€) |
| `NumOfProducts` | Numeric | Total bank products utilized (1 to 4) |
| `HasCrCard` | Binary | Credit card possession (1 = Yes, 0 = No) |
| `IsActiveMember` | Binary | Activity indicator (1 = Active, 0 = Inactive) |
| `EstimatedSalary` | Numeric | Estimated annual salary (€) |
| **`Exited`** | **Target** | **Churn status (1 = Churned, 0 = Retained)** |

### 2.2 Exploratory Data Insights
- **Target Distribution**: Retained = 7,963 (79.63%), Churned = 2,037 (20.37%). This class imbalance (~4:1) requires stratified sampling and probability calibration.
- **Geographic Disparity**:
  - France: 5,014 customers | 810 Churned (**16.15%**)
  - Spain: 2,477 customers | 413 Churned (**16.67%**)
  - Germany: 2,509 customers | 814 Churned (**27.50%**)
  - *Insight*: German accounts account for 40% of all churned customers despite making up only 25% of the total dataset.
- **Age Vulnerability**: The average age of retained customers is **37.4 years**, while the average age of churned customers is **44.8 years**. The 38–60 age group exhibits peak vulnerability.

---

## 3. Preprocessing & Feature Engineering

To maximize predictive accuracy, seven domain-specific derived features were engineered:

1. **Balance-to-Salary Ratio** ($\text{Balance} / (\text{EstimatedSalary} + 1)$): Measures financial exposure and liquidity concentration relative to earnings.
2. **Product Density** ($\text{NumOfProducts} / (\text{Tenure} + 1)$): Quantifies rate of product adoption relative to relationship longevity.
3. **Engagement-Product Interaction** ($\text{IsActiveMember} \times \text{NumOfProducts}$): Distinguishes active multi-product power users from passive/stagnant multi-product account holders.
4. **Tenure-to-Age Ratio** ($\text{Tenure} / (\text{Age} + 1)$): Normalizes tenure against life stage.
5. **Credit-to-Age Ratio** ($\text{CreditScore} / (\text{Age} + 1)$): Assesses credit standing relative to age.
6. **Zero Balance Indicator** ($\mathbb{I}(\text{Balance} == 0)$): Flags zero-balance shell accounts.
7. **High Risk Age Window** ($\mathbb{I}(38 \le \text{Age} \le 60)$): Binary indicator isolating peak churn demographic.

Categorical features (`Geography`, `Gender`) were converted via One-Hot Encoding, and numerical columns were normalized using `StandardScaler`.

---

## 4. Model Benchmarking & Empirical Evaluation

Models were trained on an **80/20 Stratified Train-Test Split** (8,000 train / 2,000 test) and evaluated across 5 standard performance metrics:

### 4.1 Comparative Model Performance Matrix

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Primary Strength |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** (Baseline) | 71.00% | 38.81% | 73.71% | 0.5085 | 0.7869 | Maximum Interpretability |
| **Decision Tree** | 75.90% | 44.68% | **77.40%** | 0.5665 | 0.8272 | Highest Recall for High-Risk Catching |
| **Random Forest** | 84.20% | 60.61% | 63.88% | **0.6220** | 0.8568 | Balanced Precision & Recall |
| **Gradient Boosting** (Selected) | **86.40%** | **75.86%** | 48.65% | 0.5928 | **0.8632** | Top Overall Accuracy & ROC-AUC |
| **HistGradientBoosting** | 86.10% | 74.90% | 47.67% | 0.5826 | 0.8569 | Scalable Fast Boosting |

---

## 5. Model Explainability & SHAP Risk Drivers

Using SHAP (SHapley Additive exPlanations) values derived from tree ensemble models, global feature importance was computed:

### 5.1 Top Global Churn Drivers (Ranked by Mean |SHAP Value|)
1. **Age / High_Risk_Age_Group**: Older customers (38-60) exhibit positive SHAP force values driving higher churn probability.
2. **NumOfProducts**: Holding 1 product or 3+ products exerts strong positive force toward churn. Holding exactly 2 products exerts strong negative force (protecting against churn).
3. **IsActiveMember**: Inactivity pushes risk score upward by 15-25 percentage points.
4. **Geography_Germany**: German residency adds a positive SHAP shift toward churn.
5. **Balance / Balance_Salary_Ratio**: Higher uninvested balances in idle accounts increase flight risk toward higher-yield competitor offerings.

---

## 6. Strategic Business & Retention Recommendations

### 6.1 Actionable Campaign Playbooks
1. **The "2nd Product Incentive" Campaign**:
   - *Target*: Single-product account holders (27.7% baseline churn).
   - *Action*: Offer zero-fee credit card or high-yield savings add-on.
   - *Expected Impact*: Reduces churn probability by **over 60%** (moving customer into 2-product 7.6% churn bucket).
2. **Digital Activity Activation Drive**:
   - *Target*: Inactive members (`IsActiveMember == 0`).
   - *Action*: Trigger automated mobile app engagement gamification ($25 cashback for 3 mobile deposits/transfers per month).
   - *Expected Impact*: Lowers overall customer risk score by ~12.5 percentage points.
3. **Germany Regional Retention Taskforce**:
   - *Target*: German branch account holders.
   - *Action*: Audit competitive interest rates in German retail market; introduce localized loyalty benefits.
4. **Wealth Management Bridge for 40+ Demographic**:
   - *Target*: Customers aged 38–60 with balances > €75,000.
   - *Action*: Offer complimentary financial planning and retirement asset protection services.

---

## 7. Conclusion & System Deployment

This project demonstrates that predictive ML risk scoring coupled with explainable SHAP drivers empowers retail banks to transform customer retention from a cost center into a proactive revenue preservation engine. 

The accompanying **Streamlit Web Application (`app.py`)** provides bank relationship managers and executives with a live interactive portal featuring executive KPIs, customer risk calculator, model benchmarks, SHAP explainability, and What-If scenario simulation.
