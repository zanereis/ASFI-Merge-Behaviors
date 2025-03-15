import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the normalized CSV file
csv_filename = "pr_summary_normalized.csv"
df = pd.read_csv(csv_filename)

# Print column names to verify format
print("Original Columns in CSV:", df.columns.tolist())

# Clean column names: strip spaces and fix parentheses formatting
df.columns = df.columns.str.strip().str.replace(" (", "(", regex=False)

# Print cleaned column names
print("Cleaned Columns in CSV:", df.columns.tolist())

# Ensure the output directory exists
plot_dir = "Normalized Plots"
os.makedirs(plot_dir, exist_ok=True)

# Mapping project states for labels
state_labels = {1: "Graduated", 2: "Retired"}
df["Project State"] = df["Project State"].map(state_labels)

# Column names for normalized PRs
normalized_pr_columns = [
    "Merged PRs(normalized)",
    "Closed PRs(normalized)",
    "Unresolved PRs(normalized)"
]

# Check for missing columns before plotting
missing_cols = [col for col in normalized_pr_columns if col not in df.columns]
if missing_cols:
    raise KeyError(f"Missing expected columns: {missing_cols}")

# Box and Violin plots for normalized PR metrics
for col in normalized_pr_columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="Project State", y=col, data=df)
    plt.title(f"Box Plot of {col}")
    plt.savefig(f"{plot_dir}/box_{col.replace(' ', '_')}.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.violinplot(x="Project State", y=col, data=df)
    plt.title(f"Violin Plot of {col}")
    plt.savefig(f"{plot_dir}/violin_{col.replace(' ', '_')}.png")
    plt.close()

# Scatter plot for Accepted PRs vs. Rejected PRs (Normalized)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df["Merged PRs(normalized)"], y=df["Closed PRs(normalized)"], hue=df["Project State"])
plt.xlabel("Accepted PRs (Normalized)")
plt.ylabel("Rejected PRs (Normalized)")
plt.title("Scatter Plot: Accepted vs. Rejected PRs (Normalized)")
plt.legend(title="Project State")
plt.savefig(f"{plot_dir}/scatter_accept_reject.png")
plt.close()

print(f"Plots saved in {plot_dir}/")
