from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train_churn_model(df):
    data = df.copy()

    # UPDATED MAP SCHEMA: Map "Churned" to 1, while tracking "No" and "Active" as 0
    data["Customer Churn Status"] = data["Customer Churn Status"].map({
        "Churned": 1, 
        "No": 0, 
        "Active": 0
    })

    # Drop rows with missing target labels (e.g., actual missing records/NaNs)
    data = data.dropna(subset=["Customer Churn Status"])

    # Features to use
    features = [
        "Age",
        "State",
        "MTN Device",
        "Gender",
        "Satisfaction Rate",
        "Customer Review",
        "Customer Tenure in months",
        "Subscription Plan",
        "Unit Price",
        "Number of Times Purchased",
        "Total Revenue",
        "Data Usage"
    ]

    target = "Customer Churn Status"

    X = data[features]
    y = data[target]

    categorical_features = [
        "State",
        "MTN Device",
        "Gender",
        "Customer Review",
        "Subscription Plan"
    ]

    numerical_features = [
        "Age",
        "Satisfaction Rate",
        "Customer Tenure in months",
        "Unit Price",
        "Number of Times Purchased",
        "Total Revenue",
        "Data Usage"
    ]

    # Preprocessing for numbers
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Preprocessing for categories
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine preprocessing
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    # Final pipeline
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 2),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 2),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 2)
    }

    return model, metrics