# Credit Risk Analytics

A portfolio project simulating a real-world credit risk analytics workflow for a commercial lending platform. Built for graduate school applications and data analyst/data scientist internship opportunities.

---

## Project Overview

This project simulates the work of a data scientist at **Horizon Lending Inc.**, a mid-sized digital consumer-lending platform. The goal is to build an end-to-end credit risk analytics pipeline — from business understanding and data exploration through model building, explainability, business simulation, and an interactive dashboard.

The project prioritizes **business value over model complexity**. Every technical decision is tied to a business outcome: reducing losses, improving portfolio profitability, or meeting regulatory requirements.

Full project plan: [`docs/project_plan.md`](docs/project_plan.md)

---

## Business Problem

Horizon Lending issues unsecured personal loans ($1,000–$35,000, 12–60 month terms) to near-prime borrowers (credit scores 580–680). This segment is underserved by traditional banks but carries elevated default risk (~8% default rate).

**Key questions this project answers:**

1. Which borrower characteristics most strongly predict default?
2. Can we build a model that outperforms simple credit-score thresholds?
3. How much money can the model save (or earn) compared to current underwriting policy?
4. Can we explain model decisions to regulators and borrowers?
5. How should the portfolio be segmented for active risk management?

**Business impact:** A 5-point improvement in AUC (0.75 → 0.80) is estimated to reduce net charge-offs by 15–20%, worth approximately $2–3 million annually for a portfolio of this size.

---

## Dataset

### Primary: Home Credit Default Risk (Kaggle Competition)

| Attribute | Detail |
|---|---|
| **Source** | [Kaggle: Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data) |
| **Size** | ~300,000 applicants, ~120 features across 8 tables |
| **Target** | `TARGET` — 1 if defaulted, 0 if repaid (8% default rate) |
| **Tables** | `application_train`, `application_test`, `bureau`, `bureau_balance`, `previous_application`, `installments_payments`, `POS_CASH_balance`, `credit_card_balance` |
| **Data richness** | Applicant demographics, external credit bureau scores, past loan repayment history, installment-level payment behavior |

### Key Feature Groups

| Group | Examples | Business Logic |
|---|---|---|
| Applicant Profile | Income, age, employment length, education, family status | Capacity and stability to repay |
| External Scores | `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3` | Third-party creditworthiness assessments |
| Bureau History | Past inquiries, outstanding debt, delinquencies across credit ecosystem | Total indebtedness and payment discipline |
| Previous Applications | Past loan amounts, statuses, days past due | Strongest predictor of future repayment behavior |
| Installment Behavior | Ratio of late payments, average days late | Granular payment discipline signal |

### Data Challenges

| Challenge | Handling Strategy |
|---|---|
| Multi-table joins | SQL-style aggregation into flat training frame |
| High missingness (>60% in some features) | Imputation + missing-indicator flags |
| Class imbalance (8% default) | Threshold tuning, class weights |
| Outliers (extreme incomes, ages) | Domain-informed capping |
| Look-ahead bias risk | Temporal train/test split verification |

---

## Project Structure

```
credit-risk-analytics/
│
├── data/
│   ├── raw/                     # Original CSV files from Kaggle
│   └── processed/               # Cleaned and feature-engineered data
│
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory data analysis
│   ├── 02_data_cleaning.ipynb   # Data cleaning and quality checks
│   ├── 03_feature_engineering.ipynb  # Feature creation and aggregation
│   ├── 04_model_building.ipynb  # Model training and hyperparameter tuning
│   ├── 05_model_evaluation.ipynb     # Rigorous evaluation and calibration
│   ├── 06_model_explainability.ipynb # SHAP, PDP, fairness analysis
│   └── 07_business_simulation.ipynb  # Profit simulation and threshold optimization
│
├── src/
│   ├── data_loader.py           # Load and cache raw data
│   ├── data_cleaner.py          # Cleaning utilities
│   ├── visualization.py         # Reusable plotting functions
│   ├── feature_engineering.py   # Feature transformation pipeline
│   ├── feature_aggregator.py    # Multi-table aggregation logic
│   ├── model.py                 # Training and prediction pipeline
│   ├── tuning.py                # Hyperparameter optimization
│   ├── evaluation.py            # Metrics and validation
│   ├── explainability.py        # SHAP and interpretability tools
│   └── business_simulation.py   # Profit and scenario analysis
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_feature_engineering.py
│   └── test_model.py
│
├── models/                      # Saved model artifacts (.pkl / .joblib)
│
├── dashboard/                   # Interactive dashboard files
│
├── reports/
│   ├── figures/                 # All exported charts
│   ├── model_card.md            # Model performance and limitations
│   ├── fairness_report.md       # Bias assessment
│   └── simulation_results.md    # Scenario comparison
│
├── docs/
│   └── project_plan.md          # Complete project blueprint
│
├── README.md                    # This file
├── requirements.txt             # Python dependencies
└── LICENSE
```

---

## Roadmap

```
Business Understanding ──> Data Cleaning ──> EDA ──> Feature Engineering
                                                          │
                                                          ▼
              Business Simulation <── Model Explainability <── Model Building
                                                                    │
                                                                    ▼
                                                           Model Evaluation
                                                                    │
                                                                    ▼
                                                              Dashboard
```

| Phase | Description | Key Output |
|---|---|---|
| **1. Business Understanding** | Frame the problem, define stakeholders, set success metrics | Problem statement, metric map |
| **2. Data Cleaning** | Load, validate, and sanitize raw data | Clean DataFrames, quality report |
| **3. EDA** | Explore patterns, risk drivers, and data quality | Visual analysis, key insights |
| **4. Feature Engineering** | Aggregate multi-table data, create domain features | 200–300 feature flat table |
| **5. Model Building** | Train, tune, and select best model (LogReg, RF, XGBoost, LightGBM) | Trained model, tuning logs |
| **6. Model Evaluation** | Rigorous holdout evaluation, calibration, stability checks | AUC, Gini, KS, calibration plot |
| **7. Model Explainability** | SHAP, partial dependence, fairness audit | Explanations, fairness report |
| **8. Business Simulation** | Translate predictions to profit, optimize threshold, compare scenarios | Profit curves, policy memo |
| **9. Dashboard** | Interactive risk portfolio explorer | Usable stakeholder dashboard |

---

## Expected Results

### Technical Outcomes

| Metric | Target | Rationale |
|---|---|---|
| AUC-ROC | ≥ 0.80 | Industry-standard discriminative power |
| Gini Coefficient | ≥ 0.60 | 2 × AUC − 1; common banking metric |
| KS Statistic | ≥ 0.45 | Separation between default and repay distributions |
| Precision at top 5% | ≥ 0.40 | Among highest-risk flagged applicants, at least 40% should actually default |
| Calibration (Brier score) | < 0.08 | Predicted probabilities must match observed frequencies |

### Business Outcomes

| Outcome | Description |
|---|---|
| **Profit-optimal threshold** | Identify the PD cutoff that maximizes total portfolio profit, not just minimizes defaults |
| **Scenario comparison** | Quantify profit lift of model-based underwriting vs. current policy |
| **Risk segmentation** | Produce 5 risk buckets (A–E) with distinct strategies for each |
| **Fairness guardrails** | Disparate impact ratio within acceptable range; explainable model |

### Portfolio Artifacts

- 7 well-documented Jupyter notebooks
- 10 reusable Python modules with unit tests
- 1 interactive stakeholder dashboard
- 3 written reports (model card, fairness, simulation)
- Full version control history (Git)

---

*Started: July 2026. This project is a work in progress. See [`docs/project_plan.md`](docs/project_plan.md) for the full design document.*

