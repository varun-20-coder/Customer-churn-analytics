"""
app.py
-------
Streamlit dashboard for the Customer Churn Analytics project.

Run with:  streamlit run app.py

Tabs:
  1. Overview     - dataset preview + key metrics
  2. SQL Insights - business queries run against SQLite
  3. EDA          - matplotlib/seaborn visualizations
  4. Predict      - live churn prediction using the trained ML model
"""

import sqlite3
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

sns.set_style('whitegrid')

st.set_page_config(page_title="Customer Churn Analytics", layout="wide")

# ---------- Load data & model (cached so it only runs once) ----------
@st.cache_data
def load_data():
    return pd.read_csv('data/customer_data.csv')


@st.cache_resource
def load_model():
    with open('models/churn_model.pkl', 'rb') as f:
        return pickle.load(f)


df = load_data()
artifact = load_model()

st.title("📊 Customer Churn Analytics Dashboard")
st.caption("A beginner-friendly end-to-end project: Pandas + SQL + Seaborn + Scikit-learn, deployed with Streamlit.")

tab1, tab2, tab3, tab4 = st.tabs(["🏠 Overview", "🗄️ SQL Insights", "📈 EDA", "🤖 Predict Churn"])

# ---------------- TAB 1: Overview ----------------
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", len(df))
    col2.metric("Churn Rate", f"{df['churn'].mean():.1%}")
    col3.metric("Avg Monthly Spend", f"₹{df['monthly_spend'].mean():.0f}")
    col4.metric("Avg Tenure (months)", f"{df['tenure_months'].mean():.1f}")

    st.markdown("""
    **About this project:** simulates a subscription/retail business that wants to know
    which customers are likely to churn (stop buying) so it can target them with
    retention offers. Data → SQL analysis → EDA → ML prediction, all in one app.
    """)

# ---------------- TAB 2: SQL Insights ----------------
with tab2:
    st.subheader("Business Insights via SQL (SQLite)")

    conn = sqlite3.connect('database/customer_data.db')

    st.markdown("**Churn rate by city**")
    q1 = """
        SELECT city, COUNT(*) AS total_customers,
               ROUND(AVG(churn) * 100, 2) AS churn_rate_pct
        FROM customers GROUP BY city ORDER BY churn_rate_pct DESC;
    """
    st.dataframe(pd.read_sql_query(q1, conn))

    st.markdown("**Average spend & tenure: churned vs retained**")
    q2 = """
        SELECT churn, ROUND(AVG(monthly_spend),2) AS avg_spend,
               ROUND(AVG(tenure_months),1) AS avg_tenure
        FROM customers GROUP BY churn;
    """
    st.dataframe(pd.read_sql_query(q2, conn))

    st.markdown("**Top 5 highest-spending customers**")
    q3 = """
        SELECT customer_id, city, monthly_spend, churn
        FROM customers ORDER BY monthly_spend DESC LIMIT 5;
    """
    st.dataframe(pd.read_sql_query(q3, conn))

    with st.expander("Try your own SQL query"):
        user_q = st.text_area("SQL query on the `customers` table:", "SELECT * FROM customers LIMIT 5;")
        if st.button("Run query"):
            try:
                st.dataframe(pd.read_sql_query(user_q, conn))
            except Exception as e:
                st.error(f"Query error: {e}")

    conn.close()

# ---------------- TAB 3: EDA ----------------
with tab3:
    st.subheader("Exploratory Data Analysis")

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(x='churn', data=df, palette='Set2', ax=ax)
        ax.set_title('Churn Distribution')
        st.pyplot(fig)

    with c2:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.boxplot(x='churn', y='monthly_spend', data=df, palette='Set2', ax=ax)
        ax.set_title('Monthly Spend by Churn Status')
        st.pyplot(fig)

    c3, c4 = st.columns(2)
    with c3:
        fig, ax = plt.subplots(figsize=(5, 4))
        numeric_df = df.select_dtypes(include='number')
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        ax.set_title('Correlation Heatmap')
        st.pyplot(fig)

    with c4:
        fig, ax = plt.subplots(figsize=(5, 4))
        city_churn = df.groupby('city')['churn'].mean().sort_values(ascending=False)
        sns.barplot(x=city_churn.index, y=city_churn.values, palette='viridis', ax=ax)
        ax.set_title('Churn Rate by City')
        ax.set_ylabel('Churn Rate')
        st.pyplot(fig)

# ---------------- TAB 4: Predict ----------------
with tab4:
    st.subheader("Predict Whether a Customer Will Churn")
    st.caption(f"Model in use: **{artifact['model_name']}** (test accuracy: {artifact['accuracy']:.1%})")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 70, 35)
        tenure_months = st.slider("Tenure (months)", 1, 60, 12)
        monthly_spend = st.slider("Monthly spend (₹)", 100, 5000, 1500)
        gender = st.selectbox("Gender", df['gender'].unique())
    with col2:
        num_purchases = st.slider("Purchases in last 3 months", 0, 30, 8)
        support_tickets = st.slider("Support tickets raised", 0, 10, 1)
        discount_used = st.selectbox("Used a discount code?", ["No", "Yes"])
        city = st.selectbox("City", df['city'].unique())

    if st.button("Predict Churn", type="primary"):
        input_df = pd.DataFrame([{
            'age': age,
            'tenure_months': tenure_months,
            'monthly_spend': monthly_spend,
            'num_purchases_last_3m': num_purchases,
            'support_tickets': support_tickets,
            'discount_used': 1 if discount_used == "Yes" else 0,
            'gender': gender,
            'city': city
        }])

        # Apply the SAME preprocessing used during training
        input_df['gender'] = artifact['le_gender'].transform(input_df['gender'])
        input_df['city'] = artifact['le_city'].transform(input_df['city'])
        input_df[artifact['numeric_cols']] = artifact['scaler'].transform(input_df[artifact['numeric_cols']])
        input_df = input_df[artifact['features']]

        pred = artifact['model'].predict(input_df)[0]
        prob = artifact['model'].predict_proba(input_df)[0][1]

        if pred == 1:
            st.error(f"⚠️ Likely to CHURN — probability: {prob:.1%}")
        else:
            st.success(f"✅ Likely to STAY — churn probability: {prob:.1%}")
