"""
db_setup.py
------------
Loads the customer CSV into a SQLite database and runs a few
business-style SQL queries. This is the "SQL" part of the project —
in an interview you can say you used SQL to pull aggregated
insights before doing any ML.
"""

import sqlite3
import pandas as pd

DB_PATH = 'database/customer_data.db'
CSV_PATH = 'data/customer_data.csv'


def load_data_to_db():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_csv(CSV_PATH)
    df.to_sql('customers', conn, if_exists='replace', index=False)
    conn.close()
    print(f"Loaded {len(df)} rows into {DB_PATH} (table: customers)")


def run_sample_queries():
    conn = sqlite3.connect(DB_PATH)

    print("\n--- Churn rate by city ---")
    q1 = """
        SELECT city,
               COUNT(*) AS total_customers,
               ROUND(AVG(churn) * 100, 2) AS churn_rate_pct
        FROM customers
        GROUP BY city
        ORDER BY churn_rate_pct DESC;
    """
    print(pd.read_sql_query(q1, conn))

    print("\n--- Avg spend: churned vs retained ---")
    q2 = """
        SELECT churn,
               ROUND(AVG(monthly_spend), 2) AS avg_spend,
               ROUND(AVG(tenure_months), 1) AS avg_tenure
        FROM customers
        GROUP BY churn;
    """
    print(pd.read_sql_query(q2, conn))

    print("\n--- Top 5 highest-spending customers ---")
    q3 = """
        SELECT customer_id, city, monthly_spend, churn
        FROM customers
        ORDER BY monthly_spend DESC
        LIMIT 5;
    """
    print(pd.read_sql_query(q3, conn))

    conn.close()


if __name__ == '__main__':
    load_data_to_db()
    run_sample_queries()
