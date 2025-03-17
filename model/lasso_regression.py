import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Create 'plots' directory if it doesn't exist
plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

# Load JSON data
file_path = "../data/monthly_data/monthly_data.json"  # Update the path if necessary
with open(file_path, "r") as file:
    data = json.load(file)

# Convert JSON to DataFrame
df = pd.DataFrame(data)

# Convert 'month' to datetime for proper sorting
df["month"] = pd.to_datetime(df["month"])

# Filter out "incubating" projects, keeping only "graduated" and "retired"
df = df[df["status"].isin(["graduated", "retired"])]

# Define numeric features
numeric_features = [
    "avg_response_time", "avg_first_response_time", "active_devs",
    "accepted_prs", "avg_time_to_acceptance", "rejected_prs",
    "avg_time_to_rejection", "unresolved_prs", "avg_thread_length",
    "new_prs", "new_comments"
]

# Normalize by total active devs to avoid division by zero
df["total_active_devs"] = df["total_active_devs"].replace(0, np.nan)
for feature in numeric_features:
    df[feature] = df[feature] / df["total_active_devs"]

# Fill missing values with 0
df[numeric_features] = df[numeric_features].fillna(0)

# Encode target variable ('graduated' = 1, 'retired' = 0)
df["status"] = df["status"].map({"graduated": 1, "retired": 0})

# Prepare X (features) and y (target)
X = df[numeric_features]
y = df["status"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Lasso regression with cross-validation
lasso_cv = LassoCV(alphas=np.logspace(-4, 4, 100), cv=5, random_state=42)
lasso_cv.fit(X_train, y_train)

# Train Lasso with the best alpha
lasso = LassoCV(alphas=[lasso_cv.alpha_])
lasso.fit(X_train, y_train)

# Extract feature importance (including zero coefficients)
feature_importance = pd.Series(lasso.coef_, index=numeric_features)

# Sort features by absolute coefficient values
feature_importance = feature_importance.sort_values(ascending=False, key=abs)

# Plot all feature coefficients (including zero)
plt.figure(figsize=(12, 6))
feature_importance.plot(kind="bar", color=['red' if val == 0 else 'blue' for val in feature_importance])
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.title("Feature Importance using Lasso Regression (Including Zero Coefficients)")
plt.xlabel("Feature")
plt.ylabel("Coefficient Value")
plt.xticks(rotation=20, ha="right")

# Save the plot
plot_path = os.path.join(plots_dir, "lasso_feature_importance.png")
plt.savefig(plot_path, bbox_inches='tight')
plt.close()

print(f"Feature importance plot saved at: {plot_path}")

# Print feature importance values
print("Feature Importance (Lasso Coefficients):")
print(feature_importance)

# Large positive coefficient → Feature strongly increases graduation probability.
# Large negative coefficient → Feature strongly decreases graduation probability.
# Coefficient near zero → Feature is either unimportant or eliminated by Lasso.
