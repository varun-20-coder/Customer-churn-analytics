"""
generate_data.py
-----------------
Creates a synthetic customer sales dataset and saves it as CSV.
In a real project you'd pull this from a company database — here we
simulate it so the project is fully reproducible.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 2000  # number of customers

# --- Basic customer info ---
customer_id = np.arange(1001, 1001 + N)
age = np.random.randint(18, 70, N)
gender = np.random.choice(['Male', 'Female'], N)
city = np.random.choice(['Delhi', 'Mumbai', 'Bangalore', 'Pune', 'Chennai'], N)

# --- Purchase behaviour ---
tenure_months = np.random.randint(1, 60, N)                     # how long they've been a customer
monthly_spend = np.round(np.random.normal(1500, 500, N), 2)     # avg monthly spend (INR)
monthly_spend = np.clip(monthly_spend, 100, None)

num_purchases = np.random.poisson(lam=8, size=N)                # purchases in last 3 months
support_tickets = np.random.poisson(lam=1.2, size=N)            # complaints/support calls
discount_used = np.random.choice([0, 1], N, p=[0.6, 0.4])       # used a discount code?

# --- Churn label (simulated with a logical rule + noise) ---
# Customers with low tenure, low spend, and many support tickets are more likely to churn.
churn_score = (
    0.6
    - 0.025 * tenure_months
    - 0.0006 * monthly_spend
    + 0.45 * support_tickets
    - 0.6 * discount_used
    + np.random.normal(0, 1, N)
)
churn_prob = 1 / (1 + np.exp(-churn_score))
churn = (churn_prob > 0.5).astype(int)

df = pd.DataFrame({
    'customer_id': customer_id,
    'age': age,
    'gender': gender,
    'city': city,
    'tenure_months': tenure_months,
    'monthly_spend': monthly_spend,
    'num_purchases_last_3m': num_purchases,
    'support_tickets': support_tickets,
    'discount_used': discount_used,
    'churn': churn
})

df.to_csv('data/customer_data.csv', index=False)
print(f"Generated {len(df)} rows -> data/customer_data.csv")
print(f"Churn rate: {df['churn'].mean():.2%}")
