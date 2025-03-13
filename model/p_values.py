import json
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Load JSON data
file_path = "../data/monthly_data/monthly_data.json"  # Adjust path if needed
with open(file_path, "r") as file:
    data = json.load(file)

# Convert JSON to DataFrame
df = pd.DataFrame(data)

# Convert 'status' to binary labels (0 = retired, 1 = graduated)
df["status"] = df["status"].map({"graduated": 1, "retired": 0})

# Drop rows with missing 'status' values
df = df.dropna(subset=["status"])

# Define numeric features
numeric_features = [
    "avg_response_time", "avg_first_response_time", "active_devs",
    "accepted_prs", "avg_time_to_acceptance", "rejected_prs",
    "avg_time_to_rejection", "unresolved_prs", "avg_thread_length", 
    "new_prs", "new_comments", "total_active_devs"
]

# Drop non-numeric columns
df = df[numeric_features + ["status"]]

# Fill NaN values with 0 (assuming missing values indicate inactivity)
df = df.fillna(0)

# Convert numeric features to categorical bins (for Chi-Square test)
for feature in numeric_features:
    df[feature] = pd.qcut(df[feature], q=4, duplicates="drop", labels=False)

# Compute p-values using Chi-Square test
p_values = {}
for feature in numeric_features:
    contingency_table = pd.crosstab(df[feature], df["status"])
    
    # Perform Chi-Square test
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
    
    # Format p-value in scientific notation
    p_values[feature] = f"{p_value:.2e}"

# Convert results to DataFrame
p_values_df = pd.DataFrame(list(p_values.items()), columns=["Feature", "p-value (scientific notation)"])

# Display results
print(p_values_df)

# Save results to a CSV file
# p_values_df.to_csv("chi_square_p_values_results.csv", index=False)




