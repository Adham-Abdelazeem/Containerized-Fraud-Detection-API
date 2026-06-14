from mlflow.models import infer_signature
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score , accuracy_score , precision_score
import joblib
import mlflow
import mlflow.sklearn
import os , sys , io


# Ensure that the output encoding is set to UTF-8 to avoid encoding issues in the terminal
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# print("1. Loading the dataset...")
# # Make sure your CSV is inside the 'data' folder
# df = pd.read_csv('data/creditcard.csv')

# # The Kaggle dataset has 'Class' as the target variable (1 = Fraud, 0 = Normal)
# # We drop the 'Time' column as it's not very helpful for a simple model
# X = df.drop(['Class', 'Time'], axis=1)
# y = df['Class']

# print("2. Splitting data into training and testing sets...")
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# print("3. Training the model...")
# # We use Logistic Regression here because it trains very fast. 
# # Remember: Our goal is to build the API, not perfect the model!
# # class_weight='balanced' helps deal with the fact that we have very few fraud cases.
# model = LogisticRegression(max_iter=1000, class_weight='balanced')
# model.fit(X_train, y_train)

# # # Let's see how well it did (Accuracy isn't the best metric for fraud, but it's okay for now)
# # accuracy = model.score(X_test, y_test)
# # print(f"Model trained! Test Accuracy: {accuracy:.4f}")

# # Let's see how well it did (Accuracy isn't the best metric for fraud, but it's okay for now)
# y_pred = model.predict(X_test)
# sensitivity = recall_score(y_test, y_pred)
# print(f"Model trained! Sensitivity (Fraud Recall): {sensitivity:.4f}")

# print("4. Saving (serializing) the model...")
# # Ensure the model directory exists
# os.makedirs('model', exist_ok=True)

# # Save the model to a file
# joblib.dump(model, 'model/fraud_model.joblib')

# print("✅ Success! Model saved to model/fraud_model.joblib")

# Building the same previous code but with MLflow tracking

# Define the mlflow endpoint (where the tracking server is running)
mlflow.set_tracking_uri("http://localhost:5000")

# Set an MLflow experiment name (this groups runs together in the MLflow UI)
mlflow.set_experiment("Fraud Detection Experiment")

# Load the dataset
print("1. Loading the dataset...")

df = pd.read_csv('data/creditcard.csv')


# The Kaggle dataset has 'Class' as the target variable (1 = Fraud, 0 = Normal)
# We drop the 'Time' column as it's not very helpful for a simple model
X = df.drop(['Class', 'Time'], axis=1)
y = df['Class']

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

    # Also save the model to a local file (for our API to load)
    print("6. Saving (serializing) the model locally...")
    os.makedirs('model', exist_ok=True)
    joblib.dump(model, 'model/fraud_model.joblib')

