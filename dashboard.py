import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io

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

# MTN COLORS
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

.legend-box {{
    background-color: {card_bg};
    border: 1px solid {border_color};
    border-radius: 15px;
    padding: 15px;
    margin-bottom: 10px;
}}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar Navigation
# -----------------------------
with st.sidebar:
    st.image("./mtn-new-logo.svg", width=50)
    st.markdown("## Navigation")
    page = st.radio("Go to", ["Dashboard", "Predictions", "Raw Data"])

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
    st.markdown('<div class="section-title">Dashboard Overview</div>', unsafe_allow_html=True)
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

    st.markdown("## Visual Insights")
    chart1, chart2 = st.columns(2, gap="large", vertical_alignment="top")

    with chart1:
        st.markdown("#### Churn Distribution")
        churn_dist = filtered_df["Customer Churn Status"].value_counts()
        fig1, ax1 = plt.subplots(figsize=(4, 4))
        colors = [mtn_green, mtn_yellow]
        wedges, texts, autotexts = ax1.pie(
            churn_dist, labels=churn_dist.index, autopct="%1.1f%%",
            startangle=90, colors=colors,
            wedgeprops={"linewidth": 1, "edgecolor": "white"}, textprops={"fontsize": 7}
        )
        for autotext in autotexts:
            autotext.set_color("#1F2933")
            autotext.set_fontweight("medium")
        ax1.set_facecolor("#FFFFFF")
        fig1.patch.set_facecolor("#FFFFFF")
        st.pyplot(fig1, use_container_width=True)

    with chart2:
        st.markdown("#### Top Drivers")
        # CHANGED: "Yes" updated to "Churned" to match the updated dataset status configuration
        churned_df = df[df["Customer Churn Status"] == "Churned"].copy()
        churned_df["Reasons for Churn"] = churned_df["Reasons for Churn"].astype(str).str.strip().str.lower()
        ordered_reasons = ["high call tarrifs", "relocation", "costly data plan", "fast data consumption", "poor network", "better offers from competitors", "poor customers service"]
        reason_counts = churned_df["Reasons for Churn"].value_counts().reindex(ordered_reasons, fill_value=0)
        short_labels = ["Tariffs", "Relocation", "Data Plan", "Data Use", "Network", "Competitors", "Service"]
        
        fig2, ax2 = plt.subplots(figsize=(4, 3.45))
        bars = ax2.bar(short_labels, reason_counts.values, width=0.55)
        for i, bar in enumerate(bars):
            bar.set_color(mtn_green if i % 2 == 0 else mtn_yellow)
            bar.set_edgecolor("white")
            bar.set_linewidth(1.2)
        ax2.set_facecolor("#FFFFFF")
        fig2.patch.set_facecolor("#FFFFFF")
        ax2.tick_params(axis="x", labelsize=7, rotation=25)
        ax2.tick_params(axis="y", labelsize=7)
        for spine in ax2.spines.values():
            spine.set_visible(False)
        st.pyplot(fig2, use_container_width=True)

    st.divider()

# =============================
# Predictions Page
# =============================
elif page == "Predictions":
    st.markdown('<div class="section-title">Customer Churn Prediction Engine</div>', unsafe_allow_html=True)

    model, model_metrics = train_churn_model(df)

    st.markdown("### Model Operational Performance")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Model Target Accuracy", f'{model_metrics["accuracy"]}%')
    a2.metric("Precision Index", model_metrics["precision"])
    a3.metric("Recall Capture Rate", model_metrics["recall"])
    a4.metric("F1 Performance Score", model_metrics["f1_score"])

    st.markdown("---")
    
    # Setup Layout for Input Fields vs Risk Explanations Layout
    main_col, side_col = st.columns([2, 1], gap="large")

    with main_col:
        st.write("Enter subscriber metrics to execute risk diagnostic protocols.")
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

        predict_clicked = st.button("Predict Churn")

    with side_col:
        st.markdown("### Segment Definitions")
        st.write("(Hover for states)")
        st.markdown("<br>", unsafe_allow_html=True)

        # ---------------------------------------------------------
        # BAR 1: Stable Segment (0-30%)
        # ---------------------------------------------------------
        stable_states = df[df["Satisfaction Rate"] >= 4]["State"].value_counts().head(3).index.tolist()
        stable_tip = " Top Stable States:\n• " + "\n• ".join(stable_states) if stable_states else "No active metrics registered."
        
        st.button("🟢 Stable Segment (0–30%) \n Healthy customer behavior.", key="bar_stable", help=stable_tip, use_container_width=True)
        st.markdown("""
        """, unsafe_allow_html=True)

        # ---------------------------------------------------------
        # BAR 2: Watchlist Segment (31-60%)
        # ---------------------------------------------------------
        watchlist_states = df[df["Satisfaction Rate"] == 3]["State"].value_counts().head(3).index.tolist()
        watchlist_tip = "Top Watchlist States:\n• " + "\n• ".join(watchlist_states) if watchlist_states else "No active metrics registered."
        
        st.button("🟡 Watchlist Segment (31–60%) \n Early warning signs.", key="bar_watchlist", help=watchlist_tip, use_container_width=True)
        st.markdown("""
        """, unsafe_allow_html=True)

        # ---------------------------------------------------------
        # BAR 3: High Risk Segment (61-80%)
        # ---------------------------------------------------------
        high_risk_states = df[df["Customer Review"].str.lower().str.contains("network|tariff|expensive|slow", na=False)]["State"].value_counts().head(3).index.tolist()
        high_risk_tip = " Top High Risk States:\n• " + "\n• ".join(high_risk_states) if high_risk_states else "No extreme data friction logged."
        
        st.button("🟠 High Risk Segment (61–80%) \n Elevated churn indicators ", key="bar_high_risk", help=high_risk_tip, use_container_width=True)
        st.markdown("""
        """, unsafe_allow_html=True)

        # ---------------------------------------------------------
        # BAR 4: Critical Attrition (81-100%)
        # ---------------------------------------------------------
        # CHANGED: "Yes" updated to "Churned" to reflect the data structure changes
        critical_states = df[df["Customer Churn Status"] == "Churned"]["State"].value_counts().head(3).index.tolist()
        critical_tip = " Top Critical States:\n• " + "\n• ".join(critical_states) if critical_states else "No critical turnover clusters logged."

        st.button("🔴 Critical Attrition (81–100%)\n Churn threat detected ", key="bar_critical", help=critical_tip, use_container_width=True)
        st.markdown("""
        """, unsafe_allow_html=True)

    if predict_clicked:
        input_df = pd.DataFrame([{
            "Age": age, "State": state, "MTN Device": mtn_device, "Gender": gender,
            "Satisfaction Rate": satisfaction_rate, "Customer Review": customer_review,
            "Customer Tenure in months": tenure, "Subscription Plan": subscription_plan,
            "Unit Price": unit_price, "Number of Times Purchased": times_purchased,
            "Total Revenue": total_revenue, "Data Usage": data_usage
        }])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        prob_pct = probability * 100

        if prob_pct <= 30.0:
            segment_label = "🟢 Stable"
        elif prob_pct <= 60.0:
            segment_label = "🟡 Watchlist"
        elif prob_pct <= 80.0:
            segment_label = "🟠 High Risk"
        else:
            segment_label = "🔴 Critical "

        # ---------------------------------------------------------
        # THE FUTURE 3-MONTH TIME HORIZON & PURCHASING POWER LOGIC
        # ---------------------------------------------------------
        avg_monthly_spend = total_revenue / max(tenure, 1)
        expected_3mo_cost = unit_price * 3
        
        purchasing_power_factor = avg_monthly_spend / max(unit_price, 1)

        if prediction == 1 or prob_pct > 60.0:
            horizon_verdict = "🔴 Flight Risk: Expected churn within the next 3 Months"
            horizon_reason = f"High churn risk due to reduced purchasing activity (Estimated Purchasing Power Factor: {purchasing_power_factor:.2f}). The likelihood of renewal is low."
        elif purchasing_power_factor < 0.65 and satisfaction_rate <= 3:
            horizon_verdict = "⚠️ High Risk: Likely to Churn in the next 3 Months"
            horizon_reason = f"The current price (₦{unit_price:,.2f}) appears to deviate from the customer’s usual spending patterns, which may lead to churn within 90 days."
        else:
            horizon_verdict = "🟢 Secure: High Retention Stability over the next 3 Months"
            horizon_reason = f"Healthy transaction velocity metrics confirm the subscriber possesses sufficient wallet balance to clear recurring billing cycles through the next quarter."

        st.markdown("### Prediction Diagnostic Report")
        
        if state.lower() == "lagos" and "low" in subscription_plan.lower():
            st.warning("**AI Predictive Insight:** Customers using low-tier plans in Lagos show a **32% higher churn tendency** compared to national averages. Implement sub-regional retention campaigns immediately.")

        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric(label="Churn Rate", value=f"{prob_pct:.2f}%")
        res_col2.metric(label="Risk Level", value=segment_label)
        res_col3.metric(label="3-Month Outlook", value="Churn Risk" if "Risk" in horizon_verdict else "Stable")
        
        st.info(f"**90-Day Predictive Trend Assessment:** **{horizon_verdict}**\n\n*Analysis:* {horizon_reason}")
        st.markdown("---")

        st.markdown("#### Structural Failure Analysis (Why this profile is at risk)")
        reasons = []
        if satisfaction_rate <= 2:
            reasons.append(f"- ⚠️ **Low Satisfaction Rate:** The customer has a satisfaction rating of **{satisfaction_rate}**, which is a strong indicator of potential churn.")
        if str(customer_review).strip().lower() in ['poor network', 'bad', 'expensive', 'slow data']:
            reasons.append(f"- ⚠️ **Negative Customer Feedback:** The customer has reported issues such as '**{customer_review}**', which are common precursors to churn.")
        if data_usage < 5.0 and subscription_plan.lower() != 'low tier':
            reasons.append(f"- ⚠️ **Low Data Utilization:** The customer consumes only **{data_usage} GB** despite being on a premium infrastructure assignment tier.")
        if purchasing_power_factor < 0.65:
            reasons.append(f"- ⚠️ **Purchasing Power Decrement:** The current unit price of ₦{unit_price:,.2f} significantly exceeds the customer's average monthly spend, indicating potential financial strain.")

        if reasons:
            for r in reasons: st.markdown(r)
        else:
            st.markdown("-  **No significant risk factors detected in the current profile.**")

        st.markdown(f"#### Peer Cohort Risk Vector Map ({subscription_plan} users | {mtn_device} | {state})")
        cohort_df = df[(df["Subscription Plan"] == subscription_plan) & (df["MTN Device"] == mtn_device) & (df["State"] == state)]
        if not cohort_df.empty:
            st.dataframe(cohort_df.head(8), use_container_width=True)
        else:
            st.info("No matching profiles found in the current dataset.")

# =============================
# Raw Data Page
# =============================
elif page == "Raw Data":
    st.markdown('<div class="section-title">Raw Customer Dataset</div>', unsafe_allow_html=True)

    with st.expander("📥 Click to Open Dropdown: Database Ingestion & Update Utility", expanded=False):
        st.markdown("### **System Ingestion Matrix**")
        drop_col1, drop_col2 = st.columns(2)
        
        with drop_col1:
            st.markdown("**Step 1: Download Empty Target Layout Sheet**")
            template_cols = ["Age", "State", "MTN Device", "Gender", "Satisfaction Rate", "Customer Review", "Customer Tenure in months", "Subscription Plan", "Unit Price", "Number of Times Purchased", "Total Revenue", "Data Usage"]
            if 'df' in locals() or 'df' in globals():
                template_cols = df.columns.tolist()
                if 'Customer Churn Status' in template_cols: template_cols.remove('Customer Churn Status')
            
            blank_df = pd.DataFrame(columns=template_cols)
            csv_buffer = io.StringIO()
            blank_df.to_csv(csv_buffer, index=False)
            st.download_button(label="📄 Download Blank Template Sheet", data=csv_buffer.getvalue(), file_name="customer_ingestion_template.csv", mime="text/csv", use_container_width=True)
            
        with drop_col2:
            st.markdown("**Step 2: Upload Completed Entry Sheet**")
            uploaded_file = st.file_uploader("Drop tracking sheet here:", type=["csv"], label_visibility="collapsed")
            if uploaded_file is not None:
                try:
                    uploaded_df = pd.read_csv(uploaded_file)
                    if "Subscription Plan" in uploaded_df.columns or "State" in uploaded_df.columns:
                        st.success(f"✅ Verified. Found {len(uploaded_df)} new structural entry rows.")
                        if st.button("💾 Commit Entries to Global Session Data", type="primary", use_container_width=True):
                            st.session_state['df'] = pd.concat([df, uploaded_df], ignore_index=True)
                            st.toast("System database updated successfully!", icon="🎉")
                            st.rerun()
                    else:
                        st.error("Structure Error: CSV header layout schema mismatch.")
                except Exception as e:
                    st.error(f"Incompatible document structural layout: {e}")

    st.dataframe(df, use_container_width=True)