import pandas as pd

def load_data():
    df = pd.read_csv("mtn_customer_churn.csv")

    # Clean column names properly
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

    # Convert date
    df["Date of Purchase"] = pd.to_datetime(
        df["Date of Purchase"],
        format="%b-%y",
        errors="coerce"
    )

    # Clean churn column
    df["Customer Churn Status"] = df["Customer Churn Status"].astype(str).str.strip()

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
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_metrics(df):
    total = df["Customer ID"].nunique()
    churned = df[df["Customer Churn Status"] == "Yes"]["Customer ID"].nunique()
    rate = round((churned / total) * 100, 2) if total > 0 else 0
    revenue = df["Total Revenue"].sum()

    return {
        "total_customers": total,
        "churned_customers": churned,
        "churn_rate": rate,
        "total_revenue": revenue
    }


def get_monthly_churn(df):
    churned = df[df["Customer Churn Status"] == "Yes"].copy()

    if churned.empty:
        return pd.Series(dtype="int")

    monthly = churned.groupby(churned["Date of Purchase"].dt.to_period("M")).size()
    monthly.index = monthly.index.astype(str)

    return monthly