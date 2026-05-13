import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from backend import load_data, get_metrics

st.set_page_config(
    page_title="MTN Customer Churn Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "theme" not in st.session_state:
    st.session_state.theme = "Light"

with st.sidebar:
    theme_choice = st.toggle("🌙 Dark Mode", value=(st.session_state.theme == "Dark"))
    st.session_state.theme = "Dark" if theme_choice else "Light"

if st.session_state.theme == "Dark":
    bg_color = "#0F0F0F"
    text_color = "#F5F5F5"
    card_bg = "#1C1C1C"
    border_color = "#333333"
    primary_color = "#FFD000"
    secondary_color = "#2B2B2B"
    muted_color = "#BBBBBB"
    chart_text = "white"
else:
    bg_color = "#FFFFFF"
    text_color = "#111111"
    card_bg = "#F7F9FA"
    border_color = "#DADADA"
    primary_color = "#00A651"
    secondary_color = "#EAF7F0"
    muted_color = "#666666"
    chart_text = "black"

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
    }}
    div[data-testid="stMetricValue"] {{
        color: {primary_color} !important;
        font-weight: 800;
    }}
</style>
""", unsafe_allow_html=True)

df = load_data()

with st.sidebar:
    st.title("Navigation")
    page = st.radio("Go to", ["📊 Dashboard", "🔍 Predictions", "📋 Raw Data"])

st.markdown(
    f"""
    <div class="hero-box">
        <div class="main-title">MTN Customer Churn Model</div>
        <div class="subtitle">
            Data-driven customer churn analysis and prediction dashboard
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

if page == "📊 Dashboard":
    st.markdown('<div class="section-title">Dashboard Overview</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

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

    chart1, chart2 = st.columns(2)

    with chart1:
        st.subheader("Churn Distribution")
        churn_dist = filtered_df["Customer Churn Status"].value_counts()
        fig1, ax1 = plt.subplots(figsize=(5, 5), facecolor=bg_color)
        colors = [primary_color, "#CFCFCF"] if len(churn_dist) > 1 else [primary_color]
        ax1.pie(
            churn_dist,
            labels=churn_dist.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            textprops={"color": chart_text}
        )
        ax1.set_facecolor(bg_color)
        st.pyplot(fig1, use_container_width=True)

    with chart2:
        st.subheader("Customer Tenure Histogram")
        fig2, ax2 = plt.subplots(figsize=(6, 5), facecolor=bg_color)
        ax2.hist(
            filtered_df["Customer Tenure in months"].dropna(),
            bins=15,
            color=primary_color,
            edgecolor="black"
        )
        ax2.set_title("Customer Tenure Distribution", color=chart_text)
        ax2.set_xlabel("Tenure in Months", color=chart_text)
        ax2.set_ylabel("Frequency", color=chart_text)
        ax2.set_facecolor(bg_color)
        ax2.tick_params(colors=chart_text)
        for spine in ax2.spines.values():
            spine.set_edgecolor(border_color)
        st.pyplot(fig2, use_container_width=True)

elif page == "🔍 Predictions":
    st.markdown('<div class="section-title">Customer Churn Prediction</div>', unsafe_allow_html=True)
    st.write("Enter customer details below to predict churn.")

    model = joblib.load("churn_model.pkl")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=10, max_value=100, value=30)
        state = st.selectbox("State", sorted(df["State"].dropna().unique()))
        mtn_device = st.selectbox("MTN Device", sorted(df["MTN Device"].dropna().unique()))
        gender = st.selectbox("Gender", sorted(df["Gender"].dropna().unique()))
        satisfaction_rate = st.slider("Satisfaction Rate", 1, 5, 3)
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
        prediction_proba = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(f"This customer is likely to churn. Probability: {prediction_proba:.2%}")
        else:
            st.success(f"This customer is likely to remain. Probability of churn: {prediction_proba:.2%}")

elif page == "📋 Raw Data":
    st.markdown('<div class="section-title">Raw Customer Dataset</div>', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True)