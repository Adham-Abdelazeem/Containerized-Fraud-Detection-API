from mlflow.models import infer_signature
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score , accuracy_score , precision_score
import joblib
import mlflow
import mlflow.sklearn
import os , sys , io
from feast import FeatureStore


# Ensure that the output encoding is set to UTF-8 to avoid encoding issues in the terminal
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Define the mlflow endpoint (where the tracking server is running)
mlflow.set_tracking_uri("http://localhost:5000")

# Set an MLflow experiment name (this groups runs together in the MLflow UI)
mlflow.set_experiment("Fraud Detection Experiment")

print("1. Loading the dataset...")

store = FeatureStore(repo_path="../feature_repo")

# Load your labels (these never went into Feast)
labels_df = pd.read_parquet("../feature_repo/data/labels.parquet")


training_df = store.get_historical_features(
    entity_df=labels_df,        # has transaction_id + event_timestamp + Class
    features=[
        "transaction_features:V1",
        "transaction_features:V2",
        # ... all features
        "transaction_features:Amount",
    ],
).to_df()



# Features + label are now in one DataFrame
X = training_df.drop(columns=["Class", "event_timestamp", "transaction_id"])
y = training_df["Class"]

# Split the data
print("2. Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Start an MLflow run (this creates a new experiment run in the MLflow UI)
with mlflow.start_run(run_name="Logistic Regression Fraud Model"):
    # Define the hyperparameters we want to track
    max_iter = 1000
    class_weight = 'balanced'

    # Log the hyperparameters to MLflow
    mlflow.log_param("max_iter", max_iter)    
    mlflow.log_param("class_weight", class_weight)

    # Train the model 
    print("3. Training the model...")
    model = LogisticRegression(max_iter=max_iter, class_weight=class_weight)
    model.fit(X_train, y_train)

    # Log evaluation metrics
    print("4. Evaluating the model...")
    y_pred = model.predict(X_test)
    sensitivity = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    F1 = 2 * (precision * sensitivity) / (precision + sensitivity)
    mlflow.log_metric("sensitivity", sensitivity)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("F1", F1)


    # Log the model itself to MLflow (this saves the model in a way that MLflow can manage)
    print("5. Logging the model to MLflow...")

    # Infering the mlflow Signature in JSON format (this captures the input and output schema of the model, which is useful for deployment later)
    signature = infer_signature(X_test, y_pred)    
    mlflow.sklearn.log_model(model,name="Logistic_Regression_Fraud",registered_model_name="Logistic_Regression_Fraud_Registered", signature=signature)


