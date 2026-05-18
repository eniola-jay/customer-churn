import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from backend import load_data, get_metrics, get_filtered_data
from churn_model import train_churn_model


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="MTN Customer Churn Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Load data
# -----------------------------
df = load_data()

bg_color = "#F7F7F5"
text_color = "#1F2933"
card_bg = "#FFFFFF"
border_color = "#D9E2EC"
primary_color = "#4F6F52"
secondary_color = "#E8EFEA"
muted_color = "#6B7280"
chart_text = "black"
sidebar_text = "#1F2933"

# MTN COLORS (NEW)
mtn_yellow = "#FFCC00"
mtn_green = "#008751"
mtn_green_soft = "#A6D6A8"
mtn_yellow_soft = "#FFE680"

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown(f"""
<style>
.stApp {{
    background-color: {bg_color};
    color: {text_color};
}}

[data-testid="stSidebar"] {{
    background-color: {card_bg};
    border-right: 1px solid {border_color};
}}

[data-testid="stSidebar"] * {{
    color: {sidebar_text} !important;
}}

.hero-box {{
    background: linear-gradient(135deg, {card_bg}, {secondary_color});
    padding: 22px;
    border-radius: 25px;
    border: 1px solid {border_color};
    margin-bottom: 18px;
}}

.main-title {{
    font-size: 32px;
    font-weight: 800;
    color: {primary_color};
}}

.subtitle {{
    font-size: 14px;
    color: {muted_color};
}}

.section-title {{
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 10px;
}}

div[data-testid="stMetric"] {{
    background-color: {card_bg};
    border: 1.5px solid #C9D7CB;
    border-radius: 20px;
    padding: 14px;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
}}

div[data-testid="stMetricLabel"] {{
    color: {muted_color} !important;
    font-size: 12px !important;
    font-weight: 400 !important;
}}

div[data-testid="stMetricValue"] {{
    color: {primary_color} !important;
    font-size: 30px !important;
    font-weight: 1000 !important;
}}

.stButton > button {{
    background-color: {primary_color};
    color: white;
    border-radius: 10px;
}}

.chart-container {{
    width: 100%;
    background: #FFFFFF;
    border-radius: 25px;
    padding: 12px;
    border: 1px solid #E5E7EB;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}}
.chart-title {{
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 8px;
    color: #1F2933;
}}

.st-emotion-cache-pa57uv img{{
    max-width: 100%;
    height: auto;
    border-radius: 20px;
    border:5px solid #E5E7EB;
}}
.st-emotion-cache-pa57uv img:hover{{
    border:5px solid #4F6F52;
    box-shadow: 0 4px 12px rgba(79, 111, 82, 0.2);
    transform: scale(1.02) translateY(-2px);
    transition: all 0.5s ease;
}}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.image(
        "./mtn-new-logo.svg",
        width=50
    )

    st.markdown("## Navigation")

    page = st.radio(
        "Go to",
        ["Dashboard", "Predictions", "Raw Data"]
    )

# -----------------------------
# Header
# -----------------------------
st.markdown(f"""
<div class="hero-box">
    <div class="main-title">MTN Customer Churn Model</div>
    <div class="subtitle">
        A responsive analytics and prediction dashboard for understanding
        customer churn behaviour and supporting retention decisions.
    </div>
</div>
""", unsafe_allow_html=True)

# =============================
# Dashboard Page
# =============================
if page == "Dashboard":

    st.markdown(
        '<div class="section-title">Dashboard Overview</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        states = st.multiselect(
            "Filter by State",
            options=sorted(df["State"].dropna().unique()),
            default=sorted(df["State"].dropna().unique())[:5]
        )

    with col2:
        plans = st.multiselect(
            "Filter by Subscription Plan",
            options=sorted(df["Subscription Plan"].dropna().unique()),
            default=sorted(df["Subscription Plan"].dropna().unique())
        )

    filtered_df = get_filtered_data(df, states, plans)
    metrics = get_metrics(filtered_df)

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Total Customers", metrics["total_customers"])
    m2.metric("Churned Customers", metrics["churned_customers"])
    m3.metric("Churn Rate", f'{metrics["churn_rate"]}%')
    m4.metric("Total Revenue", f'₦{metrics["total_revenue"]:,.0f}')

    # -----------------------------
    # Visual Insights
    # -----------------------------
    st.markdown("## Visual Insights")

    chart1, chart2 = st.columns(2, gap="large", vertical_alignment="top")


    # -----------------------------
    # Churn Distribution
    # -----------------------------
    with chart1:

        st.markdown("#### Churn Distribution")

        churn_dist = filtered_df["Customer Churn Status"].value_counts()

        fig1, ax1 = plt.subplots(figsize=(4, 3))

        colors = [mtn_green, mtn_yellow]

        wedges, texts, autotexts = ax1.pie(
            churn_dist,
            labels=churn_dist.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            wedgeprops={"linewidth": 1, "edgecolor": "white"},
            textprops={"fontsize": 7}
        )

        for autotext in autotexts:
            autotext.set_color("#1F2933")
            autotext.set_fontweight("medium")

        ax1.set_facecolor("#FFFFFF")
        fig1.patch.set_facecolor("#FFFFFF")
        ax1.set_aspect("equal")

        st.pyplot(fig1, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # Top Churn Drivers
    # -----------------------------
    with chart2:

        st.markdown("#### Top Drivers")

        churned_df = df[df["Customer Churn Status"] == "Yes"].copy()

        churned_df["Reasons for Churn"] = (
            churned_df["Reasons for Churn"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        ordered_reasons = [
            "high call tarrifs",
            "relocation",
            "costly data plan",
            "fast data consumption",
            "poor network",
            "better offers from competitors",
            "poor customers service"
        ]

        reason_counts = (
            churned_df["Reasons for Churn"]
            .value_counts()
            .reindex(ordered_reasons, fill_value=0)
        )

        short_labels = [
            "Tariffs",
            "Relocation",
            "Data Plan",
            "Data Use",
            "Network",
            "Competitors",
            "Service"
        ]

        fig2, ax2 = plt.subplots(figsize=(4, 3.75))

        bars = ax2.bar(
            short_labels,
            reason_counts.values,
            width=0.55
        )

        for i, bar in enumerate(bars):
            if i % 2 == 0:
                bar.set_color(mtn_green)
            else:
                bar.set_color(mtn_yellow)

            bar.set_edgecolor("white")
            bar.set_linewidth(1.2)

        ax2.set_facecolor("#FFFFFF")
        fig2.patch.set_facecolor("#FFFFFF")

        ax2.tick_params(axis="x", labelsize=7, rotation=25)
        ax2.tick_params(axis="y", labelsize=7)

        for spine in ax2.spines.values():
            spine.set_visible(False)

        st.pyplot(fig2, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

# =============================
# Predictions Page
# =============================
elif page == "Predictions":

    st.markdown(
        '<div class="section-title">Customer Churn Prediction</div>',
        unsafe_allow_html=True
    )

    model, model_metrics = train_churn_model(df)

    st.markdown("### Model Performance")

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Accuracy", f'{model_metrics["accuracy"]}%')
    a2.metric("Precision", model_metrics["precision"])
    a3.metric("Recall", model_metrics["recall"])
    a4.metric("F1 Score", model_metrics["f1_score"])

    st.markdown("---")
    st.write("Enter customer details below to predict churn.")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=10, max_value=100, value=30)
        state = st.selectbox("State", sorted(df["State"].dropna().unique()))
        mtn_device = st.selectbox("MTN Device", sorted(df["MTN Device"].dropna().unique()))
        gender = st.selectbox("Gender", sorted(df["Gender"].dropna().unique()))
        satisfaction_rate = st.slider("Satisfaction Rate", min_value=1, max_value=5, value=3)
        customer_review = st.selectbox("Customer Review", sorted(df["Customer Review"].dropna().unique()))

    with col2:
        tenure = st.number_input("Customer Tenure in months", min_value=0, max_value=120, value=12)
        subscription_plan = st.selectbox("Subscription Plan", sorted(df["Subscription Plan"].dropna().unique()))
        unit_price = st.number_input("Unit Price", min_value=0.0, value=5000.0)
        times_purchased = st.number_input("Number of Times Purchased", min_value=0, value=5)
        total_revenue = st.number_input("Total Revenue", min_value=0.0, value=50000.0)
        data_usage = st.number_input("Data Usage", min_value=0.0, value=20.0)

    if st.button("Predict Churn"):
        input_df = pd.DataFrame([{
            "Age": age,
            "State": state,
            "MTN Device": mtn_device,
            "Gender": gender,
            "Satisfaction Rate": satisfaction_rate,
            "Customer Review": customer_review,
            "Customer Tenure in months": tenure,
            "Subscription Plan": subscription_plan,
            "Unit Price": unit_price,
            "Number of Times Purchased": times_purchased,
            "Total Revenue": total_revenue,
            "Data Usage": data_usage
        }])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.markdown("### Prediction Result")
        if prediction == 1:
            st.error(f"Customer is likely to churn. Probability: {probability:.2%}")
        else:
            st.success(f"Customer is likely to remain. Probability of churn: {probability:.2%}")

# =============================
# Raw Data Page
# =============================
elif page == "Raw Data":

    st.markdown(
        '<div class="section-title">Raw Customer Dataset</div>',
        unsafe_allow_html=True
    )

    st.dataframe(df, use_container_width=True)