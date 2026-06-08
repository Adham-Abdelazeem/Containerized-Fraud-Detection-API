import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
import joblib
import os

print("1. Loading the dataset...")
# Make sure your CSV is inside the 'data' folder
df = pd.read_csv('data/creditcard.csv')

# The Kaggle dataset has 'Class' as the target variable (1 = Fraud, 0 = Normal)
# We drop the 'Time' column as it's not very helpful for a simple model
X = df.drop(['Class', 'Time'], axis=1)
y = df['Class']

print("2. Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("3. Training the model...")
# We use Logistic Regression here because it trains very fast. 
# Remember: Our goal is to build the API, not perfect the model!
# class_weight='balanced' helps deal with the fact that we have very few fraud cases.
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# # Let's see how well it did (Accuracy isn't the best metric for fraud, but it's okay for now)
# accuracy = model.score(X_test, y_test)
# print(f"Model trained! Test Accuracy: {accuracy:.4f}")

# Let's see how well it did (Accuracy isn't the best metric for fraud, but it's okay for now)
y_pred = model.predict(X_test)
sensitivity = recall_score(y_test, y_pred)
print(f"Model trained! Sensitivity (Fraud Recall): {sensitivity:.4f}")

print("4. Saving (serializing) the model...")
# Ensure the model directory exists
os.makedirs('model', exist_ok=True)

# Save the model to a file
joblib.dump(model, 'model/fraud_model.joblib')

print("✅ Success! Model saved to model/fraud_model.joblib")