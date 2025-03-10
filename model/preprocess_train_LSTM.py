import json
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

# Select relevant numerical features for LSTM
numeric_features = [
    "active_devs", "accepted_prs", "rejected_prs", "unresolved_prs",
    "new_prs", "new_comments", "avg_thread_length", "avg_time_to_acceptance",
    "avg_time_to_rejection"
]

# Fill NaN values with 0 (assuming missing values indicate inactivity)
df[numeric_features] = df[numeric_features].fillna(0)

# Normalize the numeric features
scaler = MinMaxScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Encode 'status' as binary labels (0 = retired, 1 = graduated)
label_encoder = LabelEncoder()
df["status"] = label_encoder.fit_transform(df["status"])

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

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Define the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, len(numeric_features))),
    #Dropout(0.2),
    LSTM(50),
    #Dropout(0.2),
    Dense(25, activation="relu"),
    Dense(1, activation="sigmoid")  # Binary classification (graduated or retired)
])

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

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
