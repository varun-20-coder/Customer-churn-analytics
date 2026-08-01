"""
train_model.py
----------------
Trains a simple, interpretable ML model (Logistic Regression) to
predict customer churn, plus a Random Forest for comparison.
Saves the best model + the fitted preprocessing objects with pickle
so app.py can load them straight away.
"""

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('data/customer_data.csv')

# --- Feature engineering ---
features = ['age', 'tenure_months', 'monthly_spend', 'num_purchases_last_3m',
            'support_tickets', 'discount_used', 'gender', 'city']
target = 'churn'

X = df[features].copy()
y = df[target]

# Encode categorical columns
le_gender = LabelEncoder()
le_city = LabelEncoder()
X['gender'] = le_gender.fit_transform(X['gender'])
X['city'] = le_city.fit_transform(X['city'])

# Scale numeric columns (helps Logistic Regression converge nicely)
scaler = StandardScaler()
numeric_cols = ['age', 'tenure_months', 'monthly_spend', 'num_purchases_last_3m', 'support_tickets']
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Model 1: Logistic Regression (simple baseline) ---
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_preds = log_model.predict(X_test)
log_acc = accuracy_score(y_test, log_preds)

# --- Model 2: Random Forest (usually a bit stronger) ---
rf_model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)

print(f"Logistic Regression accuracy: {log_acc:.3f}")
print(f"Random Forest accuracy:       {rf_acc:.3f}")

# Pick the better model
best_model, best_name = (rf_model, 'RandomForest') if rf_acc >= log_acc else (log_model, 'LogisticRegression')
print(f"\nBest model: {best_name}")
print("\nClassification report:\n", classification_report(y_test, best_model.predict(X_test)))
print("Confusion matrix:\n", confusion_matrix(y_test, best_model.predict(X_test)))

# Feature importance (only meaningful for RF, but nice to show if chosen)
if best_name == 'RandomForest':
    importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nFeature importances:\n", importances)

# --- Save everything the Streamlit app needs ---
artifact = {
    'model': best_model,
    'model_name': best_name,
    'scaler': scaler,
    'le_gender': le_gender,
    'le_city': le_city,
    'numeric_cols': numeric_cols,
    'features': features,
    'accuracy': max(log_acc, rf_acc)
}

with open('models/churn_model.pkl', 'wb') as f:
    pickle.dump(artifact, f)

print("\nSaved model artifact -> models/churn_model.pkl")
