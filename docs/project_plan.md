# Credit Risk Analytics Portfolio Project — Blueprint

> **Purpose:** Graduate school application & data analyst internship portfolio piece

---

## Table of Contents

1. [Business Background](#1-business-background)
2. [Data](#2-data)
3. [Analytics Roadmap](#3-analytics-roadmap)
4. [Deliverables](#4-deliverables)
5. [Skills Demonstrated](#5-skills-demonstrated)

---

## 1. Business Background

### 1.1 Lending Scenario

**Lender:** A mid-sized digital consumer-lending platform operating in the United Statesunsecured personal loans (USD $1,000–$35,000) with terms of 12–60 months. Borrowers apply online; decisions are made within minutes.

Horizon's portfolio currently holds ~50,000 active loans. The company targets the "near-prime" segment — borrowers with credit scores between 580 and 680 — a population that is underserved by traditional banks but carries higher default risk.

### 1.2 Why Predicting Default Is Valuable

Each percentage-point improvement in default prediction accuracy translates to millions in avoided losses. Specifically:

- **Origination losses:** Every defaulted loan erodes the principal. A model that flags high-risk applicants before funding lets the lender decline or adjust terms (higher APR, lower amount).
- **Capital reserves:** Basel III regulations require banks to hold capital against expected losses. Better prediction reduces reserve requirements.
- **Portfolio pricing:** With accurate risk segmentation, the lender can offer competitive rates to low-risk borrowers (retaining good customers) while charging risk-appropriate rates to higher-risk borrowers.
- **Collections prioritization:** Even after origination, a default-probability score helps collections teams focus resources on accounts most likely to go bad.

**Estimated business impact:** A 5-point improvement in AUC (e.g., 0.75 → 0.80) could reduce net charge-offs by 15–20%, worth ~$2–3M annually for a portfolio of this size.

### 1.3 Stakeholders

| Stakeholder | Role | Concern |
|---|---|---|
| **Chief Risk Officer (CRO)** | Owns the credit risk policy | "Is the model sound, fair, and compliant?" |
| **Underwriting Team** | Makes loan decisions | "Can we trust this score to automate decisions?" |
| **Collections Manager** | Manages delinquent accounts | "Which accounts should we call first?" |
| **Finance / Treasury** | Manages capital allocation | "What is our expected loss next quarter?" |
| **Regulatory Compliance** | Ensures fair lending practices | "Does the model discriminate against protected groups?" |
| **Product Manager** | Designs loan terms | "Can we offer better rates to good borrowers while protecting margin?" |

### 1.4 Business Objectives

1. **Primary:** Build a default-probability model (PD model) that achieves AUC ≥ 0.80 on a holdout test set.
2. **Secondary:** Translate model outputs into a dollar-denominated decision framework (expected loss, approval threshold, profit simulation).
3. **Tertiary:** Produce an interactive dashboard that lets stakeholders explore portfolio risk at a glance.
4. **Compliance signal:** Perform basic fairness analysis to surface any disparate impact by protected attributes (if available in the data).

---

## 2. Data

### 2.1 Recommended Dataset

**Dataset:** [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data) (Kaggle competition)

**Why this dataset:**

- **Real financial data** from a genuine lending company (Home Credit).
- **Large enough** (~300K applicants with ~120 features across multiple tables) to demonstrate scalable workflows.
- **Imbalanced target** (~8% default rate) — realistic for credit risk.
- **Rich feature set** includes application data, credit bureau history, previous loan repayment behavior, and external data sources.
- **Well-documented** with data dictionaries on Kaggle.
- **Train/test split is provided**, mimicking a real production setup where you cannot peek at future data.

> **Alternative fallback:** If the full Home Credit dataset is too large (~1GB compressed), use the [UCI German Credit](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)) or [UCI Taiwanese Default](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) dataset. These are smaller but less impressive for a portfolio project. **Recommendation: use Home Credit for the main pipeline and note the alternative as a fallback.**

### 2.2 Target Variable

| Column | Description | Business Meaning |
|---|---|---|
| `TARGET` | Binary: 1 = defaulted, 0 = repaid | Whether the borrower failed to repay the loan. An 8% default rate is typical for near-prime consumer lending. |

**Definition of default per Home Credit:** The borrower had a payment delay of more than X days on their first loan installment. This aligns with industry-standard "90+ days past due" definitions.

### 2.3 Key Features & Business Interpretation

Features are organized into logical groups:

#### Applicant Profile

| Feature | Description | Business Logic |
|---|---|---|
| `NAME_CONTRACT_TYPE` | Cash loan vs. revolving loan | Revolving loans (credit cards) often carry higher risk. |
| `CODE_GENDER` | Male / Female | Used for fairness analysis (not for lending decisions per Regulation B / ECOA). |
| `AMT_INCOME_TOTAL` | Applicant's annual income | Core capacity-to-pay metric. |
| `AMT_CREDIT` | Loan amount requested | Larger loans = larger exposure. |
| `AMT_ANNUITY` | Monthly installment | Debt-to-income ratio = annuity / income. |
| `NAME_EDUCATION_TYPE` | Education level | Correlates with income stability. |
| `NAME_FAMILY_STATUS` | Marital status | Proxy for household financial support. |
| `DAYS_BIRTH` | Age in days (negative, relative to application date) | Younger applicants tend to have thinner credit files. |
| `DAYS_EMPLOYED` | Days employed (negative). Positive = unemployed duration | Employment stability is a strong risk signal. |

#### External Risk Scores

| Feature | Description | Business Logic |
|---|---|---|
| `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3` | Normalized external credit scores from third-party bureaus | These are the single most predictive features — they capture the applicant's creditworthiness as assessed by external agencies. |

#### Previous Loan Behavior

| Feature Group | Description | Business Logic |
|---|---|---|
| Previous application data (multiple tables) | Past loan amounts, statuses, days past due | Historical repayment behavior is the strongest predictor of future behavior. |

#### Bureau Data

| Feature Group | Description | Business Logic |
|---|---|---|
| Credit bureau records | Number of past inquiries, outstanding debt, delinquencies across all credit lines | Captures total indebtedness and payment discipline across the entire credit ecosystem. |

### 2.4 Data Challenges

| Challenge | Why It Matters |
|---|---|
| **Multiple tables** (application, bureau, previous_application, installments, POS_CASH_balance, credit_card_balance) | Requires SQL-like joins and aggregation — demonstrates real-world data wrangling. |
| **Missing data** — some features are >60% missing (especially external sources) | Must decide: impute, flag, or drop. Business context matters (e.g., missing `EXT_SOURCE` could mean "thin file" = riskier). |
| **Class imbalance** — 8% default rate | Accuracy paradox: a naive "predict all 0" model gets 92% accuracy but is useless. |
| **Outliers** — extreme income values, unreasonable ages | Requires domain-informed capping. |
| **Feature leakage risk** — some bureau features may not be available at origination time | Must carefully time-stamp features to avoid look-ahead bias. |

---

## 3. Analytics Roadmap

Below is the end-to-end workflow. Each stage lists purpose, expected outputs, and business value.

### Stage 1: Business Understanding

**Purpose:** Frame the analytics problem in business terms. Define what "good" looks like from a risk-management perspective.

**Activities:**
- Define the credit risk policy context.
- Translate business objectives into ML metrics (AUC, precision@k, expected dollar loss).
- Establish success criteria with stakeholders.
- Plan for regulatory constraints (fair lending, explainability).

**Expected Outputs:**
- Problem statement document.
- Stakeholder-metrics mapping table (business metric ↔ ML metric).
- Success criteria checklist.

**Business Value:** Ensures the model solves the right problem. Prevents wasted effort on technically impressive but commercially irrelevant modeling.

---

### Stage 2: Data Cleaning

**Purpose:** Load raw CSV files from the Home Credit dataset, perform initial sanity checks, and produce clean DataFrames ready for analysis.

**Activities:**
- Load all tables: `application_train`, `application_test`, `bureau`, `bureau_balance`, `previous_application`, `installments_payments`, `POS_CASH_balance`, `credit_card_balance`.
- Check data types, missing-value proportions, basic ranges.
- Handle obvious errors (negative `DAYS_EMPLOYED` for employed applicants → recode as NaN).
- Standardize column naming conventions.
- Deduplicate if needed.
- Verify the train/test split is temporal (no future leaking into past).

**Expected Outputs:**
- `data/raw/` with original CSVs (documented).
- `notebooks/01_data_cleaning.ipynb` with cleaning steps.
- `src/data_loader.py` — reusable load function.
- Data quality report (missingness heatmap, zero-variance flags).

**Business Value:** Garbage in, garbage out. A model trained on dirty data will fail in production. This stage demonstrates data hygiene — a skill hiring managers prize above modeling.

---

### Stage 3: Exploratory Data Analysis (EDA)

**Purpose:** Understand patterns, relationships, and risk drivers in the data. Generate hypotheses for feature engineering.

**Activities:**
- **Univariate analysis:** Distributions of all features. Histograms for numeric, bar charts for categorical.
- **Target rate analysis:** Default rate by feature bins. For example:
  - Default rate by income bracket.
  - Default rate by `EXT_SOURCE_2` decile.
  - Default rate by loan purpose.
- **Missing-value analysis:** Are missing values systematic? Do applicants with missing `EXT_SOURCE` have higher default rates? (If so, missingness itself is informative.)
- **Correlation analysis:** Pearson correlation matrix + heatmap. Identify collinear feature groups.
- **Feature-target relationship:** Box plots, violin plots of top features split by `TARGET`.
- **Bivariate analysis:** How do features interact? (e.g., high loan amount + low external score = very high risk.)
- **Temporal EDA:** Are default rates stable over time? If not, the model may degrade.

**Expected Outputs:**
- `notebooks/02_eda.ipynb` with full visual analysis.
- `reports/figures/` — saved charts (histograms, correlation heatmap, target-rate plots).
- `src/visualization.py` — reusable plotting functions.
- Key findings summary (e.g., "EXT_SOURCE_2 is the single most predictive feature; missing EXT_SOURCE values signal higher risk").
- Feature-selection shortlist.

**Business Value:** EDA is where the analyst builds intuition. Every chart tells a story about borrower behavior. Presenting these to stakeholders builds credibility. It also prevents modeling blind alleys.

---

### Stage 4: Feature Engineering

**Purpose:** Transform raw data into predictive signals. Aggregate information across tables. Create business-relevant features.

**Activities:**
- **Aggregate bureau data:**
  - Count of past credits, total debt, average days past due, number of delinquent accounts.
  - Feature per credit type (real estate, car, credit card, etc.).
- **Aggregate previous application data:**
  - Number of previous loan applications.
  - Approval rate on previous applications.
  - Average `DAYS_DECISION` (time to decide — longer = more scrutiny = riskier?).
- **Aggregate installment/payment data:**
  - Ratio of late payments to total installments.
  - Average days late.
  - Number of installments fully paid vs. missed.
- **Create domain features:**
  - Debt-to-income ratio: `AMT_ANNUITY / AMT_INCOME_TOTAL`.
  - Loan-to-income ratio: `AMT_CREDIT / AMT_INCOME_TOTAL`.
  - Credit utilization (if revolving).
  - Employment length bin.
  - Age bin.
- **Handle missingness:**
  - Impute numeric with median (or a separate "missing" constant for missing-as-signal features).
  - Impute categorical with "Unknown".
  - Add missing-indicator flags for key features (especially `EXT_SOURCE`).
- **Encode categoricals:**
  - Ordinal encoding for ordered categories (education).
  - One-hot encoding for nominal categories (loan purpose, family status).
- **Scale numeric features:** StandardScaler or RobustScaler (preferred for credit data due to outliers).
- **Handle class imbalance:** Note strategy (will apply during modeling — e.g., SMOTE, class weights, or threshold tuning).

**Expected Outputs:**
- `notebooks/03_feature_engineering.ipynb`.
- `src/feature_engineering.py` — reusable pipeline for train and test.
- `src/feature_aggregator.py` — functions to aggregate bureau/previous tables.
- Feature store (processed DataFrame with ~200–300 features).
- Feature-importance pre-screen (e.g., mutual information scores).

**Business Value:** In credit risk, feature engineering is where domain knowledge directly improves model performance. A good engineer can lift AUC by 0.05–0.10 through smart aggregation alone. This stage proves you understand both data and lending.

---

### Stage 5: Model Building

**Purpose:** Train, tune, and select the best model for predicting default. Compare multiple algorithms and choose based on business utility, not just accuracy.

**Activities:**
- **Split strategy:** 70/15/15 train/validation/test. Use stratified splitting to preserve class balance.
- **Baseline model:** Logistic Regression (interpretable, industry-standard for credit scoring).
- **Advanced models:**
  - Random Forest (handles non-linearity, feature interactions).
  - XGBoost or LightGBM (state-of-the-art for tabular data, handles missing values natively).
  - **Optional:** MLP / TabNet for deep learning exploration (primarily for discussion).
- **Hyperparameter tuning:**
  - Grid search or Bayesian optimization (Optuna) for the winning model.
  - Tune: learning rate, max depth, subsample, colsample_bytree, regularization (L1/L2).
- **Handling class imbalance:**
  - Compare: class-weight adjustment vs. SMOTE vs. threshold-moving.
  - Use precision-recall curve to select the optimal decision threshold.
- **Cross-validation:** 5-fold stratified CV on the training set to ensure stability.
- **Model selection criteria:**
  - AUC-ROC (standard for credit risk).
  - Precision@k (top 5% highest-risk — where collections will focus).
  - Expected dollar loss at various thresholds (business-aligned metric).

**Expected Outputs:**
- `notebooks/04_model_building.ipynb`.
- `src/model.py` — training pipeline.
- `src/tuning.py` — hyperparameter optimization.
- `models/` — saved model artifacts (`.pkl` or `.joblib`).
- Leaderboard table comparing models across metrics.
- Selected model with chosen threshold.

**Business Value:** The model is the core asset. But the **process** (splitting, tuning, comparing, handling imbalance, preventing leakage) is what recruiters evaluate. A well-documented modeling pipeline shows rigor.

---

### Stage 6: Model Evaluation

**Purpose:** Evaluate the final model rigorously on the holdout test set. Go beyond AUC — assess business impact.

**Activities:**
- **ROC curve & AUC** on test set — confirm no overfitting (train AUC vs. test AUC gap < 0.03).
- **Precision-Recall curve** — more informative for imbalanced data.
- **Confusion matrix** at the selected threshold.
- **KS statistic** — Kolmogorov-Smirnov (industry-standard for credit score separation).
- **Gini coefficient** — 2 × AUC − 1 (common reporting metric in banking).
- **Calibration plot** — does predicted probability match actual default rate? (If not, apply Platt scaling or isotonic regression.)
- **Population stability analysis** — compare score distribution between train and test sets. (If distributions differ significantly, the model may not generalize.)

**Expected Outputs:**
- `notebooks/05_model_evaluation.ipynb`.
- `reports/figures/` — ROC curve, PR curve, calibration plot, KS chart.
- `reports/model_card.md` — model card summarizing performance, limitations, and intended use.
- Final metrics table (AUC, Gini, KS, Precision@5%, Recall@5%, Brier score).

**Business Value:** A model is only valuable if it works in production. Rigorous evaluation catches failures before deployment. Calibration is especially important in credit risk: if the model says 10% default probability, the bank needs to trust that means "10 out of 100 will default."

---

### Stage 7: Model Explainability

**Purpose:** Open the black box. Understand why the model makes specific predictions. Build trust with stakeholders and satisfy regulatory requirements.

**Activities:**
- **Global feature importance:**
  - SHAP summary plot (beeswarm) — which features drive predictions across the whole dataset?
  - SHAP bar plot — top 15 features by mean absolute SHAP value.
  - Partial dependence plots (PDP) for top 5 features — how does each feature affect the prediction?
- **Local interpretability:**
  - SHAP waterfall plot for a high-risk applicant — why was this person predicted to default?
  - SHAP waterfall plot for a low-risk applicant — what factors made them safe?
  - Compare two similar applicants with different predictions (contrastive explanation).
- **Fairness analysis (Regulatory Check):**
  - If `CODE_GENDER` is available, check: are approval rates significantly different by gender? Is the disparity driven by legitimate financial factors or by model bias?
  - Calculate **disparate impact ratio** = (approval rate for protected group) / (approval rate for reference group). Ratio < 0.80 is a red flag (EEOC "Four-Fifths Rule").
  - SHAP dependence plot for gender — does gender have residual explanatory power after controlling for income, credit history, etc.?
- **Business rule extraction:** Can the model be approximated by a simple scorecard (e.g., "points" system)? This bridges ML output to regulatory-friendly interpretability.

**Expected Outputs:**
- `notebooks/06_model_explainability.ipynb`.
- `reports/figures/` — SHAP plots, PDPs, fairness dashboard.
- `reports/fairness_report.md` — bias assessment and mitigation recommendations.
- `src/explainability.py` — SHAP and PDP utilities.

**Business Value:** In regulated industries (banking, insurance, healthcare), "because the model said so" is not a valid answer. Regulators demand explanations. This stage demonstrates that you can build complex models *and* explain them to non-technical stakeholders.

---

### Stage 8: Business Simulation

**Purpose:** Translate model predictions into dollar-denominated business decisions. Simulate the impact of deploying the model.

**Activities:**
- **Expected Loss calculation:**
  - `Expected Loss (EL) = Probability of Default (PD) × Loss Given Default (LGD) × Exposure at Default (EAD)`
  - Assume LGD = 60% (typical unsecured personal loan recovery rate is ~40%).
  - EAD = loan amount (`AMT_CREDIT`).
- **Profit simulation:**
  - For each applicant: `Expected Profit = (Revenue if repaid) − (Loss if defaulted)`
  - Revenue = sum of interest payments over loan term (can approximate as `AMT_ANNUITY × term_months − AMT_CREDIT`).
  - Compute profit at various approval thresholds (e.g., approve only PD < 0.05 vs. PD < 0.10 vs. PD < 0.20).
- **Threshold optimization:**
  - Plot: approval rate vs. total expected profit.
  - Identify the threshold that maximizes profit (not just minimizes default rate).
  - Show the tradeoff: strict threshold = safer but shrinks portfolio; loose threshold = grows portfolio but increases losses.
- **What-if scenarios:**
  - Scenario A: Current policy (approve everyone who meets minimum FICO ≥ 580).
  - Scenario B: Model-only (approve based on PD threshold).
  - Scenario C: Hybrid (model as override for borderline FICO cases).
  - Compare portfolio size, total expected loss, and total profit across scenarios.
- **Collections simulation:**
  - If the model can identify the top 5% highest-risk applicants before they default, what is the dollar value of early intervention?
  - Assume early intervention reduces LGD by 20%. Compute the savings.

**Expected Outputs:**
- `notebooks/07_business_simulation.ipynb`.
- `reports/simulation_results.md` — scenario comparison table.
- `reports/figures/` — profit curve, threshold optimization plot, scenario bar chart.
- Decision policy recommendation memo.

**Business Value:** This is the **bridge from data science to business strategy**. It answers the CEO's question: "What does this model mean for our bottom line?" Recruiters love to see candidates who can translate probabilities into profit.

---

### Stage 9: Dashboard

**Purpose:** Build an interactive dashboard that allows stakeholders to monitor portfolio risk and explore model outputs.

**Recommended Tool:** Tableau Public, Power BI, or Plotly Dash.

**Dashboard Pages:**

1. **Portfolio Overview**
   - Total loans, total exposure, average PD, default rate.
   - PD distribution histogram.
   - Key metrics at a glance (AUC, Gini, KS).

2. **Risk Segmentation**
   - Portfolio broken into risk buckets (e.g., A: PD < 0.02, B: 0.02–0.05, C: 0.05–0.10, D: 0.10–0.20, E: > 0.20).
   - Loan count and total exposure per bucket.
   - Drill-down: click a bucket to see individual applicants.

3. **Feature Explorer**
   - Dropdown to select a feature → displays default rate by feature bin.
   - SHAP summary embedded or referenced.

4. **What-If Simulator (if building with Dash/Shiny)**
   - Slider to adjust approval threshold.
   - Live update: approval rate, expected profit, portfolio size.

5. **Fairness Monitor**
   - Approval rates by gender/age group.
   - Disparate impact ratio.

**Expected Outputs:**
- `dashboard/` — dashboard source files or packaged workbook.
- Demo video or screenshots for portfolio.

**Business Value:** A model that sits in a Jupyter notebook has zero business impact. A dashboard puts insights in the hands of decision-makers. It also proves you have full-stack data skills — not just modeling, but deployment and communication.

---

## 4. Deliverables

### 4.1 Directory Structure

```
credit-risk-analytics/
│
├── data/
│   ├── raw/                  # Original CSV files (not committed to git if >100MB)
│   └── processed/            # Cleaned & feature-engineered data
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_building.ipynb
│   ├── 05_model_evaluation.ipynb
│   ├── 06_model_explainability.ipynb
│   └── 07_business_simulation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── data_cleaner.py
│   ├── visualization.py
│   ├── feature_engineering.py
│   ├── feature_aggregator.py
│   ├── model.py
│   ├── tuning.py
│   ├── evaluation.py
│   ├── explainability.py
│   └── business_simulation.py
│
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_feature_engineering.py
│   └── test_model.py
│
├── models/
│   └── (saved .pkl or .joblib files)
│
├── dashboard/
│   └── (dashboard source files)
│
├── reports/
│   ├── figures/               # All saved charts
│   ├── model_card.md
│   ├── fairness_report.md
│   └── simulation_results.md
│
├── docs/
│   └── PROJECT_BLUEPRINT.md   # This file
│
├── README.md                  # Project overview & instructions
├── requirements.txt           # Python dependencies
└── LICENSE
```

### 4.2 Deliverable Summary

| # | Deliverable | Format | Audience |
|---|---|---|---|
| 1 | Project Blueprint | `docs/PROJECT_BLUEPRINT.md` | Self, mentors |
| 2 | Data Cleaning Notebook | `notebooks/01_data_cleaning.ipynb` | Technical peers |
| 3 | EDA Notebook | `notebooks/02_eda.ipynb` | All stakeholders |
| 4 | Feature Engineering Notebook | `notebooks/03_feature_engineering.ipynb` | Technical peers |
| 5 | Model Building Notebook | `notebooks/04_model_building.ipynb` | Data scientists |
| 6 | Model Evaluation Notebook | `notebooks/05_model_evaluation.ipynb` | Risk team |
| 7 | Model Explainability Notebook | `notebooks/06_model_explainability.ipynb` | Compliance, CRO |
| 8 | Business Simulation Notebook | `notebooks/07_business_simulation.ipynb` | Finance, CRO |
| 9 | Reusable Python Modules | `src/*.py` | Engineering team |
| 10 | Unit Tests | `tests/*.py` | QA / engineering |
| 11 | Interactive Dashboard | `dashboard/` | Executive team |
| 12 | Model Card | `reports/model_card.md` | Compliance |
| 13 | Fairness Report | `reports/fairness_report.md` | Compliance, legal |
| 14 | Simulation Results | `reports/simulation_results.md` | Finance |
| 15 | All Charts | `reports/figures/` | Presentations |
| 16 | README | `README.md` | Portfolio reviewers |
| 17 | Requirements | `requirements.txt` | Reproducibility |

---

## 5. Skills Demonstrated

| Stage | Skills Demonstrated | Why Recruiters Care |
|---|---|---|
| **1. Business Understanding** | Stakeholder analysis, problem framing, metric definition | Separates "button-pushers" from consultants. Shows you solve problems, not just run models. |
| **2. Data Cleaning** | Data integrity, missing-value handling, type coercion | 80% of data work is cleaning. Proves production readiness. |
| **3. EDA** | Statistical thinking, data visualization, pattern recognition | Shows curiosity. Recruiters want analysts who find insights, not just generate outputs. |
| **4. Feature Engineering** | Domain knowledge, multi-table aggregation, creativity | The highest-leverage skill in applied ML. A great feature engineer is worth three model tuners. |
| **5. Model Building** | Algorithm selection, hyperparameter tuning, cross-validation, imbalance handling | Fundamental technical competency. Must-have for any DS role. |
| **6. Model Evaluation** | ROC/PR curves, calibration, KS/Gini, overfitting detection | Prevents embarrassment in production. Shows you understand *when* a model will fail. |
| **7. Model Explainability** | SHAP, partial dependence, fairness analysis, regulatory compliance | Critical for banking, insurance, healthcare. Growing regulatory scrutiny makes this a differentiator. |
| **8. Business Simulation** | P&L thinking, threshold optimization, scenario analysis, ROI communication | The #1 gap recruiters see: candidates who can build models but can't say what they're worth. |
| **9. Dashboard** | Data visualization, stakeholder communication, interactivity | Proves you can deliver value to non-technical audiences. Most DS roles require presentation skills. |
| **Cross-cutting** | Git version control, project structure, documentation, reproducibility, unit testing | Shows engineering maturity. Teams want data scientists who write maintainable code. |

---

## Appendix: Why This Project Stands Out

| Common Portfolio Projects | This Project |
|---|---|
| "Train XGBoost on Kaggle, get AUC 0.82, stop." | End-to-end workflow from business problem to profit simulation. |
| One notebook, minimal comments. | Seven notebooks + reusable Python modules + tests + dashboard. |
| No business context. | Full stakeholder map, profit simulation, policy recommendations. |
| No fairness analysis. | Dedicated fairness report for regulatory compliance. |
| Model is a black box. | Full SHAP explainability + partial dependence plots. |
| No deployment artifact. | Interactive dashboard for non-technical stakeholders. |

---

*This blueprint is a living document. As the project progresses, findings may warrant adjustments to the plan. All changes will be documented in version control.*
