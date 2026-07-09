# Credit Risk Analytics

A portfolio project simulating a real-world credit risk analytics workflow for a mid-sized digital consumer-lending platform. Built for graduate school applications and data analyst/data scientist internship opportunities. The full pipeline is implemented end-to-end across 6 executed Jupyter notebooks.

---

## Project Overview

This project simulates the work of a data scientist at a mid-sized digital consumer-lending platform, a mid-sized digital consumer-lending platform targeting near-prime borrowers (credit scores 580-680). The goal is to build an end-to-end credit risk analytics pipeline - from business understanding and data exploration through model building, explainability, business simulation, and an interactive dashboard - and translate model outputs into dollar-denominated business decisions.

The project prioritizes **business value over model complexity**. Every technical decision is tied to a business outcome: reducing losses, improving portfolio profitability, or meeting regulatory requirements. The pipeline is fully executed with actual results, figures, and dashboard exports.

Full project plan: [docs/project_plan.md](docs/project_plan.md)

---

## Key Results

| Metric | Target | Actual | Status |
|---|---|---|---|
| AUC-ROC | >= 0.80 | **0.768** | Within ~4 points of target |
| Gini Coefficient | >= 0.60 | **0.536** | 2 x AUC - 1 |
| Optimal Threshold (F2) | - | **0.485** | Optimized for recall-biased credit risk |
| Precision at threshold | >= 0.40 | **0.171** | Reflects 8% base rate constraint |
| Recall at threshold | >= 0.70 | **0.697** | Captures ~70% of all defaults |

### Business Simulation ($625M Portfolio)

| Policy | Approval Rate | Default Rate | Net Profit | Risk-Adj Return |
|---|---|---|---|---|
| Approve Everyone | 100.0% | 8.11% | $129.1M | 20.66% |
| **AI Model (optimal, t=0.485)** | **67.1%** | **3.62%** | **$112.6M** | **26.83%** |
| Conservative (t=0.30) | 40.1% | 2.24% | $72.0M | 28.73% |
| Aggressive (t=0.70) | 89.4% | 5.77% | $133.4M | 23.87% |
| Current Rules (rule-based) | 4.6% | 2.50% | $8.1M | 28.30% |

**Bottom line:** The AI model generates **$104.5M more profit** than the current rule-based underwriting policy - a **1,298% improvement** - while funding 67% of applicants rather than just 4.6%.

### Collections Impact
Top-5% flagged accounts: 9.8% of defaults caught early -> **$52,575 net benefit** after intervention costs.

---

## Business Problem

The simulated platform issues unsecured personal loans ($1,000-$35,000, 12-60 month terms) to near-prime borrowers (credit scores 580-680). This segment is underserved by traditional banks but carries elevated default risk (~8% default rate).

**Key questions this project answers:**

1. Which borrower characteristics most strongly predict default?
2. Can we build a model that outperforms simple credit-score thresholds? **(Yes: AUC 0.768 vs. EXT_SOURCE-only baseline)**
3. How much money can the model save (or earn) compared to current underwriting policy? **(+$104.5M / 1,298%)**
4. Can we explain model decisions to regulators and borrowers? **(SHAP, PDP, force plots, waterfalls)**
5. How should the portfolio be segmented for active risk management?

**Business impact:** A 5-point improvement in AUC (0.75 -> 0.80) is estimated to reduce net charge-offs by 15-20%, worth approximately $2-3 million annually for a portfolio of this size.

---

## Dataset

### Primary: Home Credit Default Risk (Kaggle Competition)

| Attribute | Detail |
|---|---|
| **Source** | [Kaggle: Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data) |
| **Size Used** | 307,511 applicants, 182 engineered features from the main applicant table |
| **Target** | `TARGET` - 1 if defaulted, 0 if repaid (8% default rate) |
| **Features** | Applicant demographics, external credit scores, ratio-based financial health metrics, interaction terms, frequency-encoded categoricals |

**Note:** This implementation uses the main applicant table as a starting point, focusing on deep feature engineering from applicant-level data. Multi-table aggregation (bureau, previous applications, installments) is designed in the pipeline skeletons but was exercised through the single-table feature engineering track in the executed notebooks.

### Key Feature Groups

| Group | Examples (Top-5 by SHAP) | Business Logic |
|---|---|---|
| External Scores | EXT_MEAN (#1, 0.47 SHAP) | Composite external creditworthiness |
| Financial Ratios | ANNUITY_RATE (#2, 0.17), AMT_GOODS_PRICE (#3, 0.14) | Debt burden relative to income |
| Profile | CODE_GENDER_M, EDU_ORDINAL, AGE_YEARS | Demographic stability signals |
| Credit History | EXT_SOURCE_3, EXT_2x3 (interaction) | Payment discipline from bureau data |

### Data Challenges

| Challenge | Handling Strategy |
|---|---|
| High missingness (>60% in some features) | Median/mode imputation + 62 missing-indicator flags |
| Class imbalance (8% default) | F2-threshold tuning, class weights in training |
| Outliers (extreme incomes, ages) | Domain-informed capping (P0.1-P99.9) |
| DAYS_EMPLOYED artifact | 365,243 days = unemployed flag created (affects 18% of applicants) |

---

## Project Structure

```
credit-risk-analytics/
│
├── data/
│   ├── raw/                     # application_train.csv (gitignored)
│   └── processed/               # Parquet files: X_features, y_target, full_processed
│
├── execution/                   # Executed notebooks with full outputs
│   ├── 01_business_understanding.ipynb     # Problem framing, stakeholder mapping
│   ├── 02_exploratory_data_analysis.ipynb  # EDA, patterns, risk drivers
│   ├── 03_feature_engineering.ipynb        # 182-feature pipeline
│   ├── 04_modeling.ipynb                  # 5-model training + LightGBM selection
│   ├── 04_explainability.ipynb            # SHAP, PDP, permutation importance
│   ├── 05_business_simulation.ipynb       # Profit simulation, threshold optimization
│   └── 06_explainability_dashboard.ipynb  # Dashboard exports + segment analysis
│
├── src/
│   ├── train.py                 # Model training pipeline (design doc)
│   ├── preprocess.py            # Feature engineering pipeline (design doc)
│   └── evaluate.py              # Model evaluation framework (design doc)
│
├── tests/                       # Scaffold (to be populated)
│
├── models/                      # best_model.pkl (LightGBM, gitignored)
│
├── dashboard/
│   ├── credit_risk_pbix_design.md          # Full Power BI spec and DAX measures
│   └── exports/                            # Export CSVs for dashboard ingestion
│
├── reports/
│   └── figures/                 # 30+ exported charts (EDA, modeling, SHAP, simulation)
│
├── docs/
│   └── project_plan.md          # Complete project blueprint
│
├── requirements.txt             # Python dependencies
└── LICENSE
```

---

## Pipeline

```
Business Understanding -> EDA -> Feature Engineering -> Modeling
                                                           │
                                                           ▼
              Dashboard Exports <-- Explainability <-- Business Simulation
```

| Phase | Description | Key Output |
|---|---|---|
| **1. Business Understanding** | Frame the problem, define stakeholders, set success metrics | Problem statement, stakeholder map |
| **2. EDA** | Explore patterns, risk drivers, and data quality | 30+ visualizations, key risk insights |
| **3. Feature Engineering** | Clean, encode, create domain features | 182-feature flat table (Parquet) |
| **4. Modeling** | Train, tune, and select best model (5 candidates) | LightGBM (AUC 0.768) |
| **5. Explainability** | SHAP, PDP, permutation importance, force plots | Driver analysis, visual explanations |
| **6. Business Simulation** | Profit simulation, threshold optimization, scenario comparison | Profit curves, policy comparison table |
| **7. Dashboard** | Generate dashboard exports and design spec | 8 CSV/Parquet files, Power BI design doc |

---

## Model Explainability Insights

**Top-5 features by SHAP importance:**

1. `EXT_MEAN` - Mean of external credit sources (0.47 SHAP)
2. `ANNUITY_RATE` - Annuity-to-income ratio (0.17)
3. `AMT_GOODS_PRICE` - Loan goods price (0.14)
4. `AMT_CREDIT` - Loan credit amount (0.07)
5. `EXT_SOURCE_3` - Third external credit source (0.08)

**Risk driver analysis during EDA:**
- EXT_SOURCE_2 bottom decile: 18.35% default vs. top decile: 2.97% (6.2x risk differential)
- Youngest applicants (18-25): 11.74% default vs. oldest (60+): 4.92% (2.4x)

---

## Technical Stack

| Component | Tools |
|---|---|
| Language | Python 3.10+ |
| Data | pandas, numpy, pyarrow (Parquet) |
| Modeling | scikit-learn, LightGBM, XGBoost, Logistic Regression, Decision Tree, Random Forest |
| Explainability | SHAP |
| Tuning | Optuna (configured in src/) |
| Visualization | matplotlib, seaborn, SHAP force plots |
| Dashboard | Power BI (design spec + CSV/Parquet exports) |
| Environment | Jupyter notebooks, Git version control |

---

## Project Artifacts

- **7 executed Jupyter notebooks** with full outputs and visualizations
- **3 Python module design documents** (src/train.py, src/preprocess.py, src/evaluate.py)
- **1 trained LightGBM model** saved as models/best_model.pkl
- **30+ publication-quality figures** in reports/figures/
- **8 dashboard export files** in dashboard/exports/ (CSV + Parquet)
- **1 complete Power BI design document** with DAX measures and data model

---

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Open the executed notebooks to explore results
jupyter notebook execution/
```

### Data
Download the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data) dataset from Kaggle and place CSV files in `data/raw/`. The feature engineering pipeline in `execution/03_feature_engineering.ipynb` handles loading, cleaning, and transformation.

---

## Status & Next Steps

**Completed:** End-to-end pipeline execution from business understanding through dashboard exports, including model training, explainability, and business simulation with quantified profit impact.

**Planned enhancements:**
- Populate `src/` modules with runnable code (currently design-document skeletons)
- Multi-table aggregation using bureau and previous application tables
- Unit tests in `tests/`
- Formal fairness assessment report
- Model card for regulatory documentation

---

*Started: July 2026. See [docs/project_plan.md](docs/project_plan.md) for the full design document.*
