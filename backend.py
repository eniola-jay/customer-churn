import pandas as pd

def load_data():
    df = pd.read_csv("mtn_customer_churn.csv")

    # Clean column names
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

    # Convert date column
    if "Date of Purchase" in df.columns:
        df["Date of Purchase"] = pd.to_datetime(
            df["Date of Purchase"],
            format="%b-%y",
            errors="coerce"
        )

    # Clean text columns
    text_cols = [
        "Customer Churn Status",
        "State",
        "MTN Device",
        "Gender",
        "Customer Review",
        "Subscription Plan"
    ]

    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Convert numeric columns
    numeric_cols = [
        "Age",
        "Satisfaction Rate",
        "Customer Tenure in months",
        "Unit Price",
        "Number of Times Purchased",
        "Total Revenue",
        "Data Usage"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_metrics(df):
    total_customers = df["Customer ID"].nunique() if "Customer ID" in df.columns else len(df)

    churned_customers = df[df["Customer Churn Status"] == "Yes"]["Customer ID"].nunique() \
        if "Customer ID" in df.columns else df[df["Customer Churn Status"] == "Yes"].shape[0]

    churn_rate = round((churned_customers / total_customers) * 100, 2) if total_customers > 0 else 0
    total_revenue = df["Total Revenue"].sum() if "Total Revenue" in df.columns else 0

    return {
        "total_customers": total_customers,
        "churned_customers": churned_customers,
        "churn_rate": churn_rate,
        "total_revenue": total_revenue
    }


def get_filtered_data(df, states=None, plans=None):
    filtered_df = df.copy()

    if states:
        filtered_df = filtered_df[filtered_df["State"].isin(states)]

    if plans:
        filtered_df = filtered_df[filtered_df["Subscription Plan"].isin(plans)]

    return filtered_df