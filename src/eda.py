"""
eda.py
-------
Basic exploratory data analysis using pandas, matplotlib, and seaborn.
Saves plots to the images/ folder so they can be reused in the
Streamlit app or a portfolio README.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

df = pd.read_csv('data/customer_data.csv')

print(df.head())
print("\nShape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nSummary stats:\n", df.describe())

# 1. Churn distribution
plt.figure(figsize=(5, 4))
sns.countplot(x='churn', data=df, palette='Set2')
plt.title('Churn Distribution (0 = Retained, 1 = Churned)')
plt.savefig('images/churn_distribution.png', bbox_inches='tight')
plt.close()

# 2. Monthly spend vs churn
plt.figure(figsize=(6, 4))
sns.boxplot(x='churn', y='monthly_spend', data=df, palette='Set2')
plt.title('Monthly Spend by Churn Status')
plt.savefig('images/spend_vs_churn.png', bbox_inches='tight')
plt.close()

# 3. Correlation heatmap (numeric columns only)
plt.figure(figsize=(7, 5))
numeric_df = df.select_dtypes(include='number')
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.savefig('images/correlation_heatmap.png', bbox_inches='tight')
plt.close()

# 4. Churn rate by city
plt.figure(figsize=(6, 4))
city_churn = df.groupby('city')['churn'].mean().sort_values(ascending=False)
sns.barplot(x=city_churn.index, y=city_churn.values, palette='viridis')
plt.ylabel('Churn Rate')
plt.title('Churn Rate by City')
plt.savefig('images/churn_by_city.png', bbox_inches='tight')
plt.close()

print("\nSaved 4 plots to images/")
