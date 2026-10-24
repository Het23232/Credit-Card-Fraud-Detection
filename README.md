# Credit Card Fraud Detection (Streamlit)

Predict fraudulent credit card transactions on a highly imbalanced dataset (~0.17% fraud), with an interactive Streamlit app.

## Dataset
Kaggle: **Credit Card Fraud Detection** (`mlg-ulb/creditcardfraud`) - https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
284,807 transactions, 492 frauds. Columns: `Time`, `V1`-`V28` (PCA-anonymised), `Amount`, `Class` (1 = fraud).

Download `creditcard.csv` and put it in this folder (or upload it from the app sidebar).
No dataset yet? Choose **Synthetic demo data** in the sidebar.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## What the app does
| Tab | Content |
|---|---|
| Data & imbalance | Class distribution, accuracy paradox, amount/hour/feature analysis |
| Model evaluation | PR curve, ROC, confusion matrix, F1, MCC, interactive threshold slider, feature importance |
| Business cost | Cost of missed fraud vs false alarms, cost-optimal threshold |
| Try transactions | Score random test transactions, or batch-score your own CSV |

## Techniques covered
- **Imbalance handling:** class weights, random undersampling, SMOTE
- **Models:** Logistic Regression (baseline), Random Forest, **Isolation Forest (unsupervised anomaly detection)**
- **Metrics:** PR-AUC (average precision), precision, recall, F1, MCC, ROC-AUC - not accuracy
- **Threshold tuning:** F1-optimal and cost-optimal thresholds
- **Feature engineering:** `Time` -> hour of day, `Amount` -> log(1 + Amount)

## Suggested report structure
1. Problem & business context  2. Data & imbalance  3. Preprocessing  4. Models & imbalance strategies
5. Results (compare PR-AUC across the 3 models x strategies)  6. Threshold & cost analysis  7. Limitations & future work (XGBoost/LightGBM, autoencoders, time-based split, explainability with SHAP)

> Note: the split here is stratified random. In production, use a time-based split to avoid leakage.
