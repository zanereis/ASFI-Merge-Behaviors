import json
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import os

# Load JSON data
file_path = "../data/monthly_data/monthly_data.json"
with open(file_path, "r") as file:
    data = json.load(file)

# Convert JSON to DataFrame
df = pd.DataFrame(data)

# Convert 'month' to datetime
df["month"] = pd.to_datetime(df["month"])

# Filter out "incubating" projects
df = df[df["status"].isin(["graduated", "retired"])]

# Select relevant numerical features
numeric_features = [
    "avg_response_time", "avg_first_response_time", "active_devs",
    "accepted_prs", "avg_time_to_acceptance", "rejected_prs",
    "avg_time_to_rejection", "unresolved_prs", "avg_thread_length", "new_prs", "new_comments"
]

# Normalize features by active devs
df["total_active_devs"] = df["total_active_devs"].replace(0, np.nan)
for feature in numeric_features:
    df[feature] = df[feature] / df["total_active_devs"]

df[numeric_features] = df[numeric_features].fillna(0)
df = df.drop(columns=["total_active_devs"])

# Normalize features
scaler = MinMaxScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Encode 'status'
df["status"] = df["status"].map({"graduated": 1, "retired": 0})

# Sort data by project and time
df = df.sort_values(by=["repo", "month"])

# Group data by project
grouped_data = df.groupby("repo")
X, y = [], []

for _, group in grouped_data:
    for i in range(len(group)):
        X.append(group.iloc[i][numeric_features].values)
        y.append(group.iloc[i]["status"])

sequence_length = 1  # Use past 3 months for prediction
X, y = [], []

for _, group in grouped_data:
    group_values = group[numeric_features].values
    for i in range(len(group_values) - sequence_length):
        X.append(group_values[i : i + sequence_length])
        y.append(group.iloc[i + sequence_length]["status"])

X = np.array(X)
y = np.array(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

# Compute class weights
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train.ravel())
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# Define the Bi-LSTM model
model = Sequential([
    Input(shape=(1, len(numeric_features))),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.2),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.2),
    Bidirectional(LSTM(64)),
    Dropout(0.2),
    Dense(25, activation="relu"),
    Dense(1, activation="sigmoid")
])

# Compile the model
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
model.fit(
    X_train, y_train, 
    epochs=30, 
    batch_size=16, 
    validation_data=(X_val, y_val), 
    callbacks=[early_stopping], 
    class_weight=class_weight_dict
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Make predictions
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob.astype("float32") > 0.5).astype(int)  # Convert probabilities to binary (0 or 1)

# Compute overall accuracy
overall_accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {overall_accuracy:.4f}")

# Compute precision, recall, F1-score, and support for each class
report = classification_report(y_test, y_pred, target_names=["Retired", "Graduated"])
print("\nClassification Report:\n", report)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", conf_matrix)

# Extract class-wise metrics
import numpy as np
report_dict = classification_report(y_test, y_pred, target_names=["Retired", "Graduated"], output_dict=True)

# Print class-wise accuracy
retired_accuracy = report_dict["Retired"]["precision"]
graduated_accuracy = report_dict["Graduated"]["precision"]

print(f"\nAccuracy for Retired Projects: {retired_accuracy:.4f}")
print(f"Accuracy for Graduated Projects: {graduated_accuracy:.4f}")


# *********************************************
# Using SHAP to interpret the prediction of the model
# Ensure SHAP can access the TensorFlow model
rng = np.random.default_rng()

# Create 'plots' directory if it doesn't exist
plots_dir = "plots/BiLSTM_SHAP"
os.makedirs(plots_dir, exist_ok=True)

# Sample background data
background = X_train[rng.choice(X_train.shape[0], 50, replace=False)]

# Ensure X_test has the correct shape
if len(X_test.shape) == 3:
    X_test_reshaped = X_test[:100].reshape(100, X_test.shape[2])  # Remove timestep dim
else:
    X_test_reshaped = X_test[:100]

# Initialize SHAP explainer
explainer = shap.GradientExplainer(model, background)

# Compute SHAP values
shap_values = explainer.shap_values(X_test[:100])

# Convert to numpy array if necessary
if isinstance(shap_values, list):
    shap_values = shap_values[0]  # Extract first class for binary classification

shap_values = np.squeeze(shap_values)  # Removes dimensions of size 1

# Ensure correct shape
print("SHAP values shape after squeeze:", shap_values.shape)
print("X_test shape:", X_test[:100].shape)

# Save summary plot
plt.figure()
shap.summary_plot(shap_values, X_test_reshaped, feature_names=numeric_features, show=False)
plt.savefig(os.path.join(plots_dir, "shap_summary_plot.png"), bbox_inches='tight')
plt.close()

# Save dependence plots
for feature in range(len(numeric_features)):
    plt.figure()
    shap.dependence_plot(feature, shap_values, X_test_reshaped, feature_names=numeric_features, show=False)
    plt.savefig(os.path.join(plots_dir, f"shap_dependence_plot_{numeric_features[feature]}.png"), bbox_inches='tight')
    plt.close()

print(f"SHAP plots saved in '{plots_dir}' folder successfully!")