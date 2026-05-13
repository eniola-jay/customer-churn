import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from backend import load_data, get_metrics, get_monthly_churn

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="MTN Customer Churn Model",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Theme State
# -----------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "Light"

theme_choice = st.sidebar.toggle(
    "🌙 Dark Mode",
    value=(st.session_state.theme == "Dark")
)
st.session_state.theme = "Dark" if theme_choice else "Light"

# -----------------------------
# Theme Colors
# -----------------------------
if st.session_state.theme == "Dark":
    bg_color = "#0F0F0F"
    text_color = "#F5F5F5"
    card_bg = "#1C1C1C"
    border_color = "#333333"
    primary_color = "#FFD000"   # yellow
    secondary_color = "#2B2B2B"
    muted_color = "#BBBBBB"
    chart_text = "white"
else:
    bg_color = "#FFFFFF"
    text_color = "#111111"
    card_bg = "#F7F9FA"
    border_color = "#DADADA"
    primary_color = "#00A651"   # green
    secondary_color = "#EAF7F0"
    muted_color = "#666666"
    chart_text = "black"

# -----------------------------
# Global CSS
# -----------------------------
st.markdown(f"""
<style>
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
    }}

    [data-testid="stSidebar"] {{
        background-color: {bg_color};
        border-right: 1px solid {border_color};
    }}

    .main-title {{
        font-size: 34px;
        font-weight: 800;
        color: {primary_color};
        margin-bottom: 4px;
    }}

    .subtitle {{
        font-size: 15px;
        color: {muted_color};
        margin-bottom: 25px;
    }}

    .hero-box {{
        background: linear-gradient(135deg, {card_bg}, {secondary_color});
        padding: 20px 24px;
        border-radius: 16px;
        border: 1px solid {border_color};
        margin-bottom: 20px;
    }}

    .section-title {{
        font-size: 20px;
        font-weight: 700;
        color: {text_color};
        margin-top: 10px;
        margin-bottom: 10px;
    }}

    div[data-testid="stMetric"] {{
        background-color: {card_bg};
        border: 1px solid {border_color};
        padding: 16px;
        border-radius: 14px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }}

    div[data-testid="stMetricLabel"] {{
        color: {muted_color} !important;
        font-weight: 600;
    }}

    div[data-testid="stMetricValue"] {{
        color: {primary_color} !important;
        font-weight: 800;
    }}

    .small-note {{
        font-size: 13px;
        color: {muted_color};
    }}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Data
# -----------------------------
df = load_data()

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/MTN_Logo.svg/800px-MTN_Logo.svg.png",
        width=90
    )
    st.markdown("## Navigation")
    page = st.radio(
        "Go to",
        ["📊 Dashboard", "🔍 Predictions", "📋 Raw Data"]
    )

# -----------------------------
# Header
# -----------------------------
st.markdown(
    f"""
    <div class="hero-box">
        <div class="main-title">MTN Customer Churn Model</div>
        <div class="subtitle">
            Data-driven customer churn analysis, retention insights, and interactive business dashboard
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Dashboard Page
# -----------------------------
if page == "📊 Dashboard":

    st.markdown('<div class="section-title">Dashboard Overview</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        states = st.multiselect(
            "Filter by State",
            sorted(df["State"].dropna().unique()),
            default=sorted(df["State"].dropna().unique())[:5]
        )

    with col2:
        plans = st.multiselect(
            "Filter by Subscription Plan",
            sorted(df["Subscription Plan"].dropna().unique()),
            default=sorted(df["Subscription Plan"].dropna().unique())
        )

    filtered_df = df.copy()

    if states:
        filtered_df = filtered_df[filtered_df["State"].isin(states)]

    if plans:
        filtered_df = filtered_df[filtered_df["Subscription Plan"].isin(plans)]

    metrics = get_metrics(filtered_df)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Customers", metrics["total_customers"])
    m2.metric("Churned Customers", metrics["churned_customers"])
    m3.metric("Churn Rate", f"{metrics['churn_rate']}%")
    m4.metric("Total Revenue", f"₦{metrics['total_revenue']:,}")

    st.markdown('<div class="section-title">Customer Churn Visuals</div>', unsafe_allow_html=True)

    chart1, chart2 = st.columns(2)

    # Pie Chart
    with chart1:
        st.markdown("#### Churn Distribution")
        churn_dist = filtered_df["Customer Churn Status"].value_counts()

        fig1, ax1 = plt.subplots(figsize=(5, 5), facecolor=bg_color)
        colors = [primary_color, "#CFCFCF"] if len(churn_dist) > 1 else [primary_color]

        ax1.pie(
            churn_dist,
            labels=churn_dist.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            textprops={"color": chart_text, "fontsize": 11}
        )
        ax1.set_facecolor(bg_color)
        st.pyplot(fig1, use_container_width=True)

    # Histogram
    with chart2:
        st.markdown("#### Customer Tenure Distribution")
        fig2, ax2 = plt.subplots(figsize=(6, 5), facecolor=bg_color)
        ax2.hist(
            filtered_df["Customer Tenure in months"],
            bins=15,
            color=primary_color,
            edgecolor="black"
        )
        ax2.set_title("Histogram of Customer Tenure", color=chart_text)
        ax2.set_xlabel("Tenure in Months", color=chart_text)
        ax2.set_ylabel("Frequency", color=chart_text)
        ax2.set_facecolor(bg_color)
        ax2.tick_params(colors=chart_text)
        for spine in ax2.spines.values():
            spine.set_edgecolor(border_color)
        st.pyplot(fig2, use_container_width=True)

    st.markdown("#### Quick Insight")
    churn_yes = filtered_df[filtered_df["Customer Churn Status"] == "Yes"].shape[0]
    churn_no = filtered_df[filtered_df["Customer Churn Status"] == "No"].shape[0]

    if churn_yes > churn_no:
        st.error("The filtered data shows more customers are churning than retained. Retention action is needed.")
    else:
        st.success("The filtered data shows more retained customers than churned customers.")

# -----------------------------
# Predictions Page
# -----------------------------
elif page == "🔍 Predictions":
    st.markdown('<div class="section-title">Prediction Support View</div>', unsafe_allow_html=True)
    st.info("This section highlights customers likely to be at risk based on available business indicators.")

    risk_df = df[
        (df["Satisfaction Rate"] <= 2) |
        (df["Customer Review"].isin(["Poor", "Fair"]))
    ][[
        "Full Name",
        "State",
        "Subscription Plan",
        "Satisfaction Rate",
        "Customer Review",
        "Customer Tenure in months",
        "Total Revenue",
        "Customer Churn Status"
    ]]

    st.dataframe(risk_df, use_container_width=True)

    st.markdown('<p class="small-note">Tip: later, this page can be connected to your trained ML model for live churn prediction.</p>', unsafe_allow_html=True)

# -----------------------------
# Raw Data Page
# -----------------------------
elif page == "📋 Raw Data":
    st.markdown('<div class="section-title">Raw Customer Dataset</div>', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True)