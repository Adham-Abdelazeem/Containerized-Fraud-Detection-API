import pandas as pd

df = pd.read_csv(".\data\creditcard.csv")

# Create a unique ID per transaction
df["transaction_id"] = range(len(df))

# Convert Time (seconds) to real timestamps
df["event_timestamp"] = (
    pd.Timestamp("2024-01-01", tz="UTC") +
    pd.to_timedelta(df["Time"], unit="s")
)

# Drop the label — it does NOT belong in the feature store
# Labels are for training, features are for serving
# Your model never receives 'Class' as input
df_features = df.drop(columns=["Class", "Time"])

# Save as Parquet
df_features.to_parquet("./feature_repo/feature_repo/data/transactions.parquet", index=False)

# Save labels separately — you'll join them manually during training
df_labels = df[["transaction_id", "event_timestamp", "Class"]]
df_labels.to_parquet("./feature_repo/feature_repo/data/labels.parquet", index=False)