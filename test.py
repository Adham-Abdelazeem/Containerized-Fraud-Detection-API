
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# 1. Print current tracking URI to verify where it's looking
print("Current Tracking URI:", mlflow.get_tracking_uri())

# 2. List all registered models to see if yours is there
print("Registered Models:")
for rm in client.search_registered_models():
    print(f"- {rm.name}")