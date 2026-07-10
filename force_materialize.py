from feast import FeatureStore
from datetime import datetime

store = FeatureStore(repo_path="./feature_repo/feature_repo")

# Define a wide time window that guarantees we catch your 2024 data
start_date = datetime(2023, 1, 1)
end_date = datetime(2026, 12, 31)

print(f"Materializing data from {start_date.date()} to {end_date.date()}...")

# Use .materialize() instead of .materialize_incremental()
store.materialize(start_date=start_date, end_date=end_date)

print("Materialization complete! Your 2024 data is now in the online store.")