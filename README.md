# Transaction Anomaly Detection — Unsupervised Fraud Risk Flagging

Flagging suspicious transactions in a dataset with **no fraud labels** — the
scenario a bank actually starts from before any confirmed fraud exists to
train against. Built with Isolation Forest, validated by injecting synthetic
anomalies with known properties, and translated into a review-budget decision
a fraud team could actually act on.

## Key results
- Reviewing the riskiest **5%** of transactions catches **62%** of injected
  synthetic anomalies; the riskiest **10%** catches **71%**.
- A simple rule-based flag (z-score + login attempts + amount/balance ratio)
  agrees with the model on only ~40% of top-flagged cases — evidence the two
  approaches catch genuinely different things, not the same thing twice.
- Three plausible features were tested and **dropped** after checking they
  carried no real signal in this data (see *Data quality findings* below) —
  rather than assumed useful because they sounded reasonable.

## Data quality findings
Checked three columns before trusting them as features:
| Column | Looked like | Actually was | Decision |
|---|---|---|---|
| `PreviousTransactionDate` | prior-transaction timestamp | dataset export timestamp (clustered in a 6-min window) | excluded |
| `DeviceID` reuse | device-fingerprint signal | "new device" on 99.5% of transactions | excluded |
| `TransactionDate` hour | time-of-day signal | constrained to a 16:00–18:00 window only | kept, flagged as low-information |

## Approach
1. Load & inspect — 2,512 transactions, 495 accounts, no fraud label
2. Data-quality checks (above) before any feature engineering
3. Feature engineering — per-account amount z-score, amount-to-balance ratio
4. EDA — amount, login-attempt, and channel distributions
5. Isolation Forest — unsupervised anomaly scoring
6. Rule-based score — transparent baseline for comparison
7. **Synthetic anomaly injection** — since no ground truth exists, 45
   transactions with known anomalous properties (amount spikes, login
   spikes, both) were injected to validate detection honestly
8. Optional Gen AI extension — LLM-generated plain-English flag explanations
   (scoped, requires an API key to run)

## Tech stack
Python · pandas · scikit-learn (IsolationForest) · matplotlib/seaborn

## Repo structure
```
.
├── README.md
├── requirements.txt
├── notebooks/
│   └── Transaction_Anomaly_Detection.ipynb
└── data/
    └── bank_transactions_data_2.csv
```

## Running it
```bash
pip install -r requirements.txt
jupyter notebook notebooks/Transaction_Anomaly_Detection.ipynb
```

## Data source
[Bank Transaction Dataset for Fraud Detection](https://www.kaggle.com/datasets/valakhorasani/bank-transaction-dataset-for-fraud-detection)
— Vala Khorasani, Kaggle, Apache 2.0 license.

## Next steps
Apply the same pipeline where per-account history runs deeper than this
dataset's ~5 transactions/account average (a thin baseline for the z-score
feature); add a supervised layer (XGBoost/LightGBM) once even a small set of
confirmed fraud cases exists to train against.
