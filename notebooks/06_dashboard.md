# Dashboard — Credit Risk Analytics

> **File:** `dashboard/credit_risk.pbix` (Power BI)
> **Design spec:** `dashboard/credit_risk_pbix_design.md`
> **Target Audience:** CRO, CFO, CEO, Head of Underwriting, Head of Collections
> **Data Source:** Processed feature table + model predictions
> **Refresh Cadence:** Daily automated

---

## Overview

A 5-page Power BI executive dashboard that translates the credit risk model into actionable business intelligence. Each page answers a specific management question with KPIs, charts, and interactive controls.

---

## Page 1: Executive Summary

**Question:** *"How risky is our portfolio right now?"*

| Element | Detail |
|---|---|
| **KPIs** | Total Loans (40,500), Approval Rate (81.0%), Default Rate (2.8%), Net Profit ($72.3M), Risk-Adj Return (14.28%), Expected Loss Rate (2.38%) |
| **Charts** | Profit vs. Threshold (line), Policy Comparison (bar), Risk Bucket Composition (stacked bar) |
| **Controls** | Threshold slider (0.01–0.50), Date range, Loan type filter |
| **Table** | Top 10 Riskiest Applicants with PD, income, DTI, EXT_SOURCE_2, risk drivers |

All KPIs update in real time as the threshold slider is dragged.

---

## Page 2: Risk Segmentation

**Question:** *"Which customer groups have the highest default risk?"*

| Chart | Purpose |
|---|---|
| Default Rate by Age Group | Identify age-risk patterns |
| Default Rate by Income Quartile | Check income-risk relationship |
| Default Rate by External Score Decile | Validate EXT_SOURCE_2 decay |
| Default Rate by Employment Length | Assess job stability impact |

**Risk Archetypes Table:**

| Segment | Proportion | Default Rate | Strategy |
|---|---|---|---|
| Thin File | 15% | 15% | Manual review |
| Overextended | 20% | 18% | Decline or lower amount |
| Past Defaulter | 8% | 35% | Auto-decline |
| Young & Unstable | 12% | 12% | Co-signer required |
| Low Score | 25% | 20% | Higher APR |
| Safe Profile | 20% | 2% | Fast-track approval |

---

## Page 3: Model Explainability

**Question:** *"What factors contribute most to default?"*

| Visual | Interactivity |
|---|---|
| SHAP Feature Importance (horizontal bar) | Sort toggle |
| SHAP Beeswarm | Hover for detail |
| Partial Dependence Plot | Feature dropdown selector |
| Individual Explanation Card | Applicant ID search |

**Dynamic insight text:** Auto-generated narrative of top 3 risk drivers and their portfolio impact.

---

## Page 4: What-If Simulator

**Question:** *"How much money can we save if we change the threshold?"*

| Control | Options |
|---|---|
| Threshold slider | 0.01 to 0.50 (step 0.005) |
| Economic scenario | Base Case, Mild Recession (+20% PD), Severe Recession (+50% PD), Boom (−15% PD) |

**Real-time KPI updates** with delta arrows vs. optimal baseline. Includes "Reset to Optimal" bookmark button.

---

## Page 5: Collections & Fairness

**Question:** *"Which accounts should we call first? Is the model fair?"*

| Collections KPIs | Fairness KPIs |
|---|---|
| 40% of defaults caught in top 5% | DIR (Age < 25): 0.89 — Amber |
| LGD reduced 20% (60% → 48%) | DIR (Unemployed): 0.66 — Red |
| Net savings: $529K | DIR (Age 50+): 0.99 — Green |
| 2,025 accounts flagged | Alert bar with traffic-light status |

**Table:** Top 20 accounts to call (ranked by PD score, actionable for collections team).

---

## Data Model

| Table | Grain | Refresh |
|---|---|---|
| `applications` | 1 row per applicant | Daily |
| `features` | 1 row per applicant | Daily |
| `thresholds` | 1 row per threshold value | Weekly |
| `rules_filter` | 1 row per rule | Weekly |
| `risk_buckets` | 1 row per bucket | Weekly |
| `collections` | 2 rows (with/without AI) | Weekly |
| `fairness` | 1 row per group-value | Monthly |

**Measures (DAX):** Total Applications, Approval Rate, Default Rate, Net Profit, Risk-Adj Return, Expected Loss Rate, Precision@5%, Loans Funded, Defaulted Loans.

---

## Implementation Order

1. **Page 1** — Core KPIs + threshold slider (most-used page)
2. **Page 4** — Slider logic + economic scenarios
3. **Page 2** — Dimension-based risk segmentation
4. **Page 3** — Pre-computed SHAP visuals (most complex build)
5. **Page 5** — Collections ranking + fairness metrics

---

## Full Project Pipeline

```
01_eda.ipynb                   — Exploratory data analysis
02_feature_engineering.ipynb   — Feature design document
03_modeling.ipynb              — Model training & evaluation design
04_explainability.ipynb        — SHAP, xAI, + Model Monitoring Framework
05_business_simulation.ipynb   — Profit simulation for management
06_dashboard.md                — Power BI dashboard design summary
```

---

*All 6 stages of the credit risk analytics project are now designed. The project is ready for implementation.*
