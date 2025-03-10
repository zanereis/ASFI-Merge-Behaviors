import json
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Remove outliers using IQR before normalization
def remove_outliers(df, features):
    for feature in features:
        upper_bound = df[feature].quantile(0.95)  # 95th percentile
            
        # Remove values above the 95th percentile
        df = df[(df[feature] >= 0) & (df[feature] <= upper_bound)]
    return df

# Load JSON data
file_path = "data/monthly_data/monthly_data.json"  # Change to the correct path
with open(file_path, "r") as file:
    data = json.load(file)

# Convert JSON to DataFrame
df = pd.DataFrame(data)

# Convert 'month' to datetime for proper sorting
df["month"] = pd.to_datetime(df["month"])

# Filter out "incubating" projects, keeping only "graduated" and "retired"
df = df[df["status"].isin(["graduated", "retired"])]

# Count the total number of records
total_records = len(df)

# # Count the number of unique projects in each category
# graduated_projects = df[df["status"] == "graduated"]["repo"].nunique()
# retired_projects = df[df["status"] == "retired"]["repo"].nunique()

# # Count the total number of records for each category
# graduated_records = df[df["status"] == "graduated"].shape[0]
# retired_records = df[df["status"] == "retired"].shape[0]

# # Print the results
# print(f"Total Records: {total_records}")
# print(f"Number of Graduated Projects: {graduated_projects}, Total Graduated Records: {graduated_records}")
# print(f"Number of Retired Projects: {retired_projects}, Total Retired Records: {retired_records}")


# Select relevant numerical features for LSTM
numeric_features = [
    "accepted_prs", "rejected_prs", "unresolved_prs",
    "new_prs", "new_comments", "avg_thread_length", "avg_time_to_acceptance",
    "avg_time_to_rejection"
]


# Normalize features by active devs
df["active_devs"] = df["active_devs"].replace(0, np.nan)  # Avoid division by zero
for feature in numeric_features:
    df[feature] = df[feature] / df["active_devs"]  # Divide by active devs

# Fill NaN values with 0 (assuming missing values indicate inactivity)
df[numeric_features] = df[numeric_features].fillna(0)

# Drop active devs
df = df.drop(columns=["active_devs"])

# Apply outlier removal before normalization
df = remove_outliers(df, numeric_features)

# Normalize the numeric features
scaler = MinMaxScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Encode 'status' as binary labels (0 = retired, 1 = graduated)
df["status"] = df["status"].map({"graduated": 1, "retired": 0})


# Sort data by project and time
df = df.sort_values(by=["repo", "month"])

# Group data by project
grouped_data = df.groupby("repo")

# Convert each project's data into sequences for LSTM
sequence_length = 12  # Using past 12 months for prediction
X, y = [], []

for _, group in grouped_data:
    if len(group) > sequence_length:
        for i in range(len(group) - sequence_length):
            X.append(group.iloc[i : i + sequence_length][numeric_features].values)
            y.append(group.iloc[i + sequence_length]["status"])

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

graduated_sequences = np.sum(y == 1)
retired_sequences = np.sum(y == 0)

print(f"Total Graduated Sequences: {graduated_sequences}")
print(f"Total Retired Sequences: {retired_sequences}")

# # Compute the total number of sequences for each class
# graduated_sequences = np.sum(y == 1)
# retired_sequences = np.sum(y == 0)

# # Print the results
# print(f"Total Sequences After Conversion - Graduated: {graduated_sequences}")
# print(f"Total Sequences After Conversion - Retired: {retired_sequences}")
# print(f"Total Sequences After Conversion: {len(y)}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Compute class weights
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# Print computed class weights
print(f"Computed Class Weights: {class_weight_dict}")

# # Compute the number of graduated and retired projects in train, validation, and test sets
# train_graduated = np.sum(y_train == 1)
# train_retired = np.sum(y_train == 0)

# val_graduated = np.sum(y_val == 1)
# val_retired = np.sum(y_val == 0)

# test_graduated = np.sum(y_test == 1)
# test_retired = np.sum(y_test == 0)

# # Display the counts
# print("\nData Distribution:")
# print(f"Train - Graduated: {train_graduated}, Retired: {train_retired}")
# print(f"Validation - Graduated: {val_graduated}, Retired: {val_retired}")
# print(f"Test - Graduated: {test_graduated}, Retired: {test_retired}")


# Define the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, len(numeric_features))),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(25, activation="relu"),
    Dense(1, activation="sigmoid")  # Binary classification (graduated or retired)
])

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Define early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',  # Metric to monitor
    patience=5,          # Number of epochs to wait for improvement
    restore_best_weights=True  # Restore the best weights after stopping
)

# Train the model with early stopping and class weights
model.fit(
    X_train, y_train, 
    epochs=50, 
    batch_size=16, 
    validation_data=(X_val, y_val), 
    callbacks=[early_stopping],  # Add early stopping callback
    class_weight=class_weight_dict  # Apply class weights
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Accuracy for the gradted and retired project 
# Make predictions
# Precision: Measures how many of the predicted "graduated" (or "retired") were actually correct.
# Recall: Measures how many actual "graduated" (or "retired") were correctly predicted.
# F1-score: A balance of Precision and Recall.
# Support: number of actual instances of each class in the test dataset
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary (0 or 1)

# Compute overall accuracy
overall_accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {overall_accuracy:.4f}")

# Compute accuracy, precision, recall, and F1-score for each class
report = classification_report(y_test, y_pred, target_names=["Retired", "Graduated"])
print("\nClassification Report:\n", report)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", conf_matrix)
