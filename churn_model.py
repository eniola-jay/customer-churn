import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("mtn_customer_churn.csv")

# Clean column names
df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

# Clean target
df["Customer Churn Status"] = df["Customer Churn Status"].astype(str).str.strip()
df["Customer Churn Status"] = df["Customer Churn Status"].map({"Yes": 1, "No": 0})

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

# Select features
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

X = df[features]
y = df[target]

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

# Preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])

# Model pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(model, "churn_model.pkl")
print("\nModel saved successfully as churn_model.pkl")