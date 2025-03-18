import subprocess
import re
import numpy as np

# Path to your Python script
script_path = "preprocess_train_BiGRU.py"  # Change this to your actual script path

# Number of times to run the script
num_runs = 5

# Initialize lists to store the metrics
accuracy_list = []
precision_grad_list = []
precision_ret_list = []
recall_grad_list = []
recall_ret_list = []
f1_grad_list = []
f1_ret_list = []
overall_precision_list = []
overall_recall_list = []
overall_f1_list = []

# Regex patterns to extract values
patterns = {
    "accuracy": r"Overall Accuracy: ([0-9\.]+)",
    "precision_grad": r"Graduated\s+([0-9\.]+)\s+[0-9\.]+\s+[0-9\.]+\s+\d+",
    "precision_ret": r"Retired\s+([0-9\.]+)\s+[0-9\.]+\s+[0-9\.]+\s+\d+",
    "recall_grad": r"Graduated\s+[0-9\.]+\s+([0-9\.]+)\s+[0-9\.]+\s+\d+",
    "recall_ret": r"Retired\s+[0-9\.]+\s+([0-9\.]+)\s+[0-9\.]+\s+\d+",
    "f1_grad": r"Graduated\s+[0-9\.]+\s+[0-9\.]+\s+([0-9\.]+)\s+\d+",
    "f1_ret": r"Retired\s+[0-9\.]+\s+[0-9\.]+\s+([0-9\.]+)\s+\d+",
    "overall_precision": r"Precision:\s+([0-9\.]+)",
    "overall_recall": r"Recall:\s+([0-9\.]+)",
    "overall_f1": r"F1-Score:\s+([0-9\.]+)"
}

# Run the script multiple times and extract results
for i in range(num_runs):
    print(f"Running iteration {i+1}...")
    result = subprocess.run(["python3", script_path], capture_output=True, text=True)
    output = result.stdout
    
    # Extract values using regex
    extracted_values = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            extracted_values[key] = float(match.group(1))
        else:
            extracted_values[key] = None  # Handle missing values
    
    # Append values to respective lists
    accuracy_list.append(extracted_values["accuracy"])
    precision_grad_list.append(extracted_values["precision_grad"])
    precision_ret_list.append(extracted_values["precision_ret"])
    recall_grad_list.append(extracted_values["recall_grad"])
    recall_ret_list.append(extracted_values["recall_ret"])
    f1_grad_list.append(extracted_values["f1_grad"])
    f1_ret_list.append(extracted_values["f1_ret"])
    overall_precision_list.append(extracted_values["overall_precision"])
    overall_recall_list.append(extracted_values["overall_recall"])
    overall_f1_list.append(extracted_values["overall_f1"])

# Compute average values
metrics = {
    "Accuracy": np.mean(accuracy_list),
    "Precision (Graduated)": np.mean(precision_grad_list),
    "Precision (Retired)": np.mean(precision_ret_list),
    "Recall (Graduated)": np.mean(recall_grad_list),
    "Recall (Retired)": np.mean(recall_ret_list),
    "F1-Score (Graduated)": np.mean(f1_grad_list),
    "F1-Score (Retired)": np.mean(f1_ret_list),
    "Overall Precision": np.mean(overall_precision_list),
    "Overall Recall": np.mean(overall_recall_list),
    "Overall F1-Score": np.mean(overall_f1_list)
}

# Save results to a file
output_file = "bi_gru_results.txt"
with open(output_file, "w") as f:
    f.write("Model Evaluation Results (Averaged over 5 Runs)\n")
    f.write("=" * 50 + "\n")
    for key, value in metrics.items():
        f.write(f"{key}: {value:.4f}\n")

print(f"Results saved to {output_file}")
