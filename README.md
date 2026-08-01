# Customer Churn Analytics Dashboard

An end-to-end beginner data analytics project: synthetic data → SQL analysis →
EDA → machine learning → deployed as an interactive Streamlit app.

## Problem statement
A retail/subscription business wants to know **which customers are likely to
stop buying (churn)** so it can proactively offer discounts or support to
retain them.

## Tech stack
| Layer | Tool |
|---|---|
| Data generation | NumPy, Pandas |
| Storage / querying | SQLite + SQL |
| Exploratory analysis | Matplotlib, Seaborn |
| Modeling | Scikit-learn (Logistic Regression, Random Forest) |
| Deployment | Streamlit |

## Folder structure
```
customer-churn-analytics/
├── data/
│   ├── generate_data.py      # creates the synthetic dataset
│   └── customer_data.csv     # generated data (2000 customers)
├── database/
│   ├── db_setup.py           # loads CSV into SQLite + sample SQL queries
│   └── customer_data.db      # generated SQLite database
├── src/
│   ├── eda.py                # matplotlib/seaborn plots -> images/
│   └── train_model.py        # trains + saves the ML model
├── models/
│   └── churn_model.pkl       # trained model + preprocessing objects
├── images/                   # saved EDA plots (png)
├── app.py                    # Streamlit app (the deliverable)
├── requirements.txt
└── README.md
```

## How to run it yourself
```bash
pip install -r requirements.txt

# 1. generate the dataset
python data/generate_data.py

# 2. load it into SQLite
python database/db_setup.py

# 3. run EDA (optional, saves plots)
python src/eda.py

# 4. train the model
python src/train_model.py

# 5. launch the dashboard
streamlit run app.py
```

## What the app does
- **Overview tab** – dataset preview and key business metrics (churn rate,
  average spend, tenure).
- **SQL Insights tab** – runs real SQL queries against SQLite (churn rate by
  city, spend comparison, top customers) and even lets you type your own
  query.
- **EDA tab** – churn distribution, spend vs churn boxplot, correlation
  heatmap, churn rate by city — all via Seaborn.
- **Predict tab** – enter a hypothetical customer's details and get a live
  churn prediction with probability, using the trained model.

## Deploying on Streamlit Community Cloud
1. Push this folder to a GitHub repo (make sure `data/customer_data.csv`,
   `database/customer_data.db`, and `models/churn_model.pkl` are committed,
   or add a startup step that regenerates them).
2. Go to share.streamlit.io, connect the repo, set `app.py` as the entry
   point, and deploy.

## How to explain this in an interview
- **Why synthetic data?** Keeps the project fully reproducible without
  needing a real company's data — but the logic (churn correlates with low
  tenure, low spend, more support tickets) mirrors real-world patterns.
- **Why SQL alongside Pandas?** Shows you can query data where it lives
  (a database) rather than always loading everything into memory — a
  realistic workflow at most companies.
- **Why two models?** Logistic Regression is simple/interpretable (good
  baseline); Random Forest usually captures non-linear patterns better. The
  script automatically picks whichever scores higher — a basic form of
  model selection you can talk through.
- **Model quality:** ~79% accuracy on holdout test data. Recall on churners
  is lower than precision — worth mentioning you could improve this with
  more features, class-balancing (e.g. SMOTE), or hyperparameter tuning.
- **Why Streamlit?** Fast way to turn a notebook-style analysis into a
  shareable, interactive tool — good for showing stakeholders (or
  interviewers) results without needing them to read code.

## Possible extensions to mention if asked "what would you improve?"
- Use a real dataset (e.g. Telco Customer Churn from Kaggle)
- Add cross-validation and hyperparameter tuning (GridSearchCV)
- Handle class imbalance explicitly (SMOTE, class_weight='balanced')
- Add SHAP values for model explainability
- Containerize with Docker for deployment
