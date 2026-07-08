# Credit Risk Analytics — Power BI Executive Dashboard Design

> **Dashboard File:** `dashboard/credit_risk.pbix`
> **Target Audience:** CRO, CFO, CEO, Head of Underwriting, Head of Collections
> **Data Source:** Processed feature table + model predictions (from `data/processed/`)
> **Refresh Cadence:** Daily automated

---

## Overview

A 5-page Power BI dashboard that translates the credit risk model outputs into actionable business intelligence. Each page answers a specific management question.

---

## Data Model (Power BI Backend)

### Tables Required

| Table | Source | Key Columns | Grain |
|---|---|---|---|
| `applications` | Model predictions output | `SK_ID_CURR`, `predicted_PD`, `actual_default`, `approved`, `loan_amount`, `income`, `age`, ... | 1 row per applicant |
| `features` | Feature-engineered table | `SK_ID_CURR`, all feature columns, SHAP values | 1 row per applicant |
| `thresholds` | Business simulation output | `threshold`, `approval_rate`, `default_rate`, `net_profit`, `loss_rate`, `risk_adj_return` | 1 row per threshold value |
| `rules_filter` | Current policy simulation | `rule_name`, `applicants_filtered`, `remaining`, `default_rate_filtered`, `default_rate_remaining` | 1 row per rule |
| `risk_buckets` | Prediction decile bins | `bucket_label`, `pd_range`, `applicant_count`, `default_rate`, `avg_loan_amount` | 1 row per risk bucket |
| `collections` | Collections simulation | `scenario`, `defaults_caught`, `lgd_before`, `lgd_after`, `savings`, `cost`, `net_benefit` | 2 rows (with/without AI) |
| `fairness` | Fairness analysis | `group_name`, `group_value`, `approval_rate`, `default_rate`, `disparate_impact_ratio` | 1 row per group-value |

### Relationships

```
applications (SK_ID_CURR) 1──* features (SK_ID_CURR)
thresholds (threshold)          ← slicer table (no relationship)
```

All other tables are disconnected and used for static display or slicer filtering.

### Measures (DAX)

```
Total Applications    = COUNTROWS(applications)
Approval Rate         = DIVIDE(COUNTROWS(FILTER(applications, applications[approved]=1)), Total Applications)
Default Rate          = DIVIDE(COUNTROWS(FILTER(applications, applications[actual_default]=1)), Total Applications)
Predicted Default Rate = AVERAGE(applications[predicted_PD])
Total Principal       = SUM(applications[loan_amount])
Total Revenue         = [Loans Funded] * 2142   (performing loan revenue)
Total Loss            = [Defaulted Loans] * 10617  (loss per default)
Net Profit            = [Total Revenue] - [Total Loss]
Risk Adj Return       = DIVIDE([Net Profit], [Total Principal])
Expected Loss Rate    = DIVIDE([Total Loss], [Total Principal])
Loans Funded          = COUNTROWS(FILTER(applications, applications[approved]=1))
Defaulted Loans       = COUNTROWS(FILTER(applications, applications[actual_default]=1))
Precision at 5%       = VAR top5 = CALCULATETABLE(TOPN(0.05*Total Applications, applications, applications[predicted_PD], DESC), ALL(applications)) RETURN DIVIDE(COUNTROWS(FILTER(top5, top5[actual_default]=1)), COUNTROWS(top5))
```

---

## Page 1: Executive Summary (CRO / CEO / Board)

**Question answered:** *"How risky is our portfolio right now, and are we on track?"*

### Layout (Top to Bottom)

```
┌────────────────────────────────────────────────────────────────────┐
│  HEADER: "Horizon Lending — Portfolio Risk Dashboard"  |  Date    │
├──────────┬──────────┬──────────┬──────────┬──────────┬────────────┤
│   KPI    │   KPI    │   KPI    │   KPI    │   KPI    │   KPI      │
│ Total    │ Approval │ Default  │ Net      │ Risk-Adj │ Expected   │
│ Loans    │ Rate     │ Rate     │ Profit   │ Return   │ Loss Rate  │
│ 40,500   │ 81.0%    │ 2.8%     │ $72.3M   │ 14.28%   │ 2.38%      │
├──────────┴──────────┴──────────┴──────────┴──────────┴────────────┤
│                                                                     │
│  CHART: Portfolio Profit vs. Threshold (Line Chart)                 │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │  $75M ┌───────────────────┐                             │       │
│  │       │                   │  Optimal                    │       │
│  │  $65M ┘                   └─────────────────            │       │
│  │  $55M                                          ────────┘       │
│  │      0.05  0.10  0.15  0.20  0.25  0.30  0.35  0.40           │
│  └─────────────────────────────────────────────────────────┘       │
│  SLICER: Threshold slider (0.01 to 0.50) — drag to update all KPIs │
├─────────────────────────────────────────────────────────────────────┤
│  CHART: Policy Comparison (Clustered Bar)     |  CHART: Portfolio  │
│  Net Profit by Policy                          |  Composition by   │
│  ┌────────────────────┐   ┌──────────────────┐  |  Risk Bucket     │
│  │ Approve All   ██   |   │ A (safe)   ██████│  |  (Stacked Bar)   │
│  │ Current      ████  |   │ B          ████  │  |                  │
│  │ AI Optimal   █████ |   │ C (border) ██    │  |                  │
│  └────────────────────┘   │ D (risky)  ░░    │  |                  │
│                           │ E (high)   ░     │  |                  │
│                           └──────────────────┘  |                  │
├─────────────────────────────────────────────────────────────────────┤
│  TABLE: Top 10 Riskiest Applicants (for quick reference)            │
│  ┌──────┬──────────┬────────┬────────┬──────────┬──────────────┐   │
│  │ ID   │ PD Score │ Income │ DTI    │ EXT_SRC2 │ Risk Drivers │   │
│  ├──────┼──────────┼────────┼────────┼──────────┼──────────────┤   │
│  │ 4521 │ 0.42     │ $32K   │ 0.46   │ 0.21     │ Low score,...│   │
│  │ ...  │ ...      │ ...    │ ...    │ ...      │ ...          │   │
│  └──────┴──────────┴────────┴────────┴──────────┴──────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### KPIs (Card Visuals)

| KPI | Value | Conditional Formatting |
|---|---|---|
| Total Loans Approved | 40,500 | None (volume metric) |
| Approval Rate | 81.0% | Green if > 80%, Amber 70-80%, Red < 70% |
| Default Rate | 2.8% | Green if < 3%, Amber 3-5%, Red > 5% |
| Net Portfolio Profit | $72.3M | Green if trending up (sparkline) |
| Risk-Adjusted Return | 14.28% | Green if > 13%, Amber 10-13%, Red < 10% |
| Expected Loss Rate | 2.38% | Green if < 2.5%, Amber 2.5-4%, Red > 4% |

### Charts

| Chart | Type | X-Axis | Y-Axis | Purpose |
|---|---|---|---|---|
| Profit vs. Threshold | Line | Threshold (0.01–0.50) | Net Profit ($) | Show the inverted-U profit curve; mark the optimal threshold |
| Policy Comparison | Clustered Bar | Policy Name | Net Profit ($) | Compare Approve All vs. Current Rules vs. AI Optimal |
| Risk Bucket Composition | Stacked Bar | Policy Name | % of Portfolio | Show distribution across risk buckets (A–E) |
| Top 10 Riskiest | Table | — | — | Drill-down to individual high-risk applicants |

### Page-Level Filters (Slicers)

- **Threshold slider:** Numeric range slider (0.01 to 0.50, step 0.005). Updates ALL KPIs and charts dynamically.
- **Date range:** Date slicer for time-based filtering (daily/weekly view).
- **Loan type:** Dropdown (Cash Loan, Revolving, All).

---

## Page 2: Risk Segmentation (Head of Underwriting)

**Question answered:** *"Which customer groups have the highest default risk?"*

### Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│  HEADER: "Risk Segmentation — Default Rate by Customer Profile"     │
├─────────────────────────────┬───────────────────────────────────────┤
│  CHART: Default Rate by     │  CHART: Default Rate by Income        │
│  Age Group (Bar Chart)       │  Quartile (Bar Chart)                 │
│  ┌──────────────────┐       │  ┌──────────────────┐                │
│  │ <25   ██████ 12% │       │  │ Q1 (low) ██████ 14%│               │
│  │ 25-35 ████   8%  │       │  │ Q2       ████   8% │               │
│  │ 35-50 ██     4%  │       │  │ Q3       ██     4% │               │
│  │ 50-65 █      2%  │       │  │ Q4 (high)█     2% │               │
│  │ 65+   █      2%  │       │  └──────────────────┘                │
│  └──────────────────┘       │                                       │
├─────────────────────────────┼───────────────────────────────────────┤
│  CHART: Default Rate by     │  CHART: Default Rate by Employment    │
│  External Score Decile      │  Length (Bar Chart)                    │
│  ┌──────────────────┐       │  ┌──────────────────┐                │
│  │ Decile 1  ██████ │       │  │ <1yr   ██████ 14%│               │
│  │ Decile 2  ████   │       │  │ 1-3yr  ████   9% │               │
│  │ ...               │       │  │ 3-10yr ██     5% │               │
│  │ Decile 10 █      │       │  │ 10yr+  █      2% │               │
│  └──────────────────┘       │  └──────────────────┘                │
├─────────────────────────────┴───────────────────────────────────────┤
│  TABLE: Risk Profile by Segment                                     │
│  ┌───────────┬──────────┬─────────┬────────┬─────────┬────────────┐│
│  │ Segment   │ # Apps   │ Default │ Avg PD │ Avg Loan│ Strategy   ││
│  ├───────────┼──────────┼─────────┼────────┼─────────┼────────────┤│
│  │ Thin File │ 7,500    │ 15%     │ 0.18   │ $8,200  │ Manual     ││
│  │ Overext.  │ 10,000   │ 18%     │ 0.22   │ $15,000 │ Decline    ││
│  │ Past Def. │ 4,000    │ 35%     │ 0.38   │ $11,000 │ Auto-Decl. ││
│  │ Young     │ 6,000    │ 12%     │ 0.14   │ $7,500  │ Co-signer  ││
│  │ Low Score │ 12,500   │ 20%     │ 0.25   │ $13,000 │ Higher APR ││
│  │ Safe      │ 10,000   │ 2%      │ 0.04   │ $14,000 │ Fast-Track ││
│  └───────────┴──────────┴─────────┴────────┴─────────┴────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

### KPIs

| KPI | Value |
|---|---|
| Highest Default Rate by Segment | 35% (Past Defaulters) |
| Lowest Default Rate by Segment | 2% (Safe Profile) |
| Risk Bucket with Most Applicants | Low Score (25%) |
| % of Portfolio in High-Risk Buckets (D+E) | 33% |

### Charts

| Chart | Type | Purpose |
|---|---|---|
| Default Rate by Age Group | Column | Identify age-related risk patterns |
| Default Rate by Income Quartile | Column | Check income-risk relationship |
| Default Rate by External Score Decile | Column | Validate EXT_SOURCE_2 decay curve |
| Default Rate by Employment Length | Column | Assess job stability impact |
| Risk Segment Profile Table | Matrix | Full detail with strategy recommendation |

### Page-Level Filters

- **Segment selector:** Multi-select dropdown (Thin File, Overextended, Past Defaulter, etc.)
- **Age range slider**
- **Income range slider**

---

## Page 3: Model Explainability & Feature Impact (Risk Analytics / Compliance)

**Question answered:** *"What factors contribute most to default? Can we explain every decision?"*

### Layout

```
┌──────────────────────────────────────────────────────────────────────┐
│  HEADER: "Model Explainability — What Drives Risk"                  │
├──────────────────────────────────────────────────────────────────────┤
│  CHART: SHAP Feature Importance (Horizontal Bar)                     │
│  ┌─────────────────────────────────────────────┐                    │
│  │ EXT_SOURCE_2     █████████████████████ 0.035│                    │
│  │ DTI              ███████████████      0.025 │                    │
│  │ PREV_DEFAULT_RT  ████████████         0.020 │                    │
│  │ LTI              ████████             0.015 │                    │
│  │ AGE_YEARS        ██████               0.012 │                    │
│  │ EMP_LENGTH       ████                 0.009 │                    │
│  └─────────────────────────────────────────────┘                    │
├──────────────────────────────────────────────────────────────────────┤
│  CHART: SHAP Beeswarm (Summary Plot)                                 │
│  ┌─────────────────────────────────────────────┐                    │
│  │ EXT_SOURCE_2  ●●●●●●●●●●●●●●●               │                    │
│  │              ○○○○○○○○○○○                     │                    │
│  │ DTI             ●●●●●●●●                     │                    │
│  │               ○○○○○○○                        │                    │
│  │ ...                                          │                    │
│  │   (interactive: hover for detail)             │                    │
│  └─────────────────────────────────────────────┘                    │
├─────────────────────────────────┬───────────────────────────────────┤
│  CHART: Partial Dependence      │  INDIVIDUAL EXPLANATION CARD      │
│  Plot (Line Chart)              │  ┌──────────────────────┐         │
│  ┌─────────────────┐            │  │ Applicant #4521      │         │
│  │ Risk ▲           │            │  │ PD: 0.42 (HIGH)      │         │
│  │      │    ┌────┐ │            │  │                      │         │
│  │      │━━━━┘    └─│            │  │ EXT_SOURCE_2: +0.15 │         │
│  │      └───────────│            │  │ DTI:          +0.09 │         │
│  │      0.2  0.4    │            │  │ PREV_DEFAULT: +0.07 │         │
│  │           DTI    │            │  │ LTI:          +0.04 │         │
│  └─────────────────┘            │  │ AGE:          +0.03 │         │
│  Dropdown: Select feature        │  └──────────────────────┘         │
│  to view its PDP                 │  Dropdown: Select applicant ID   │
├─────────────────────────────────┴───────────────────────────────────┤
│  KPI: Top 3 Risk Drivers (Dynamic Text Box)                          │
│  "The top 3 drivers of default risk are EXT_SOURCE_2 (external      │
│   credit score), DTI (debt-to-income ratio), and PREV_DEFAULT_RATE  │
│   (past default history). These account for 63% of total SHAP       │
│   importance across the portfolio."                                  │
└──────────────────────────────────────────────────────────────────────┘
```

### KPIs

| KPI | Value | Interpretation |
|---|---|---|
| Top Feature | EXT_SOURCE_2 | External credit score dominates |
| Top-3 Features Share | 63% | Of total SHAP importance |
| Features with Positive Direction | 5 of top 6 | Higher value = higher risk for DTI, LTI, PREV_DEFAULT |
| Features with Negative Direction | 1 of top 6 | Lower value = higher risk for EXT_SOURCE_2, AGE |

### Charts

| Chart | Type | Interactive Element | Purpose |
|---|---|---|---|
| SHAP Feature Importance | Horizontal Bar | Sort toggle (by mean abs SHAP) | Rank risk drivers |
| SHAP Beeswarm | Custom Scatter | Hover for detail | Show direction + distribution |
| Partial Dependence Plot | Line | Feature dropdown | Show marginal effect |
| Individual Explanation Card | Card + Table | Applicant ID dropdown | Local explanation |

### Page-Level Filters

- **Applicant ID:** Dropdown search box (type ID to see their explanation)
- **Feature selector:** Dropdown to choose which PDP to display

---

## Page 4: What-If Simulator (CEO / CFO / Strategy)

**Question answered:** *"How much money can we save by changing the approval threshold? What happens to profit if the economy changes?"*

### Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│  HEADER: "What-If Simulator — Policy Threshold & Economic Impact"   │
├─────────────────────────────────────────────────────────────────────┤
│  SLICER PANEL (Left Side)                                           │
│  ┌────────────────────────────────────────┐                        │
│  │  DECISION THRESHOLD                    │                        │
│  │  ┌─────────────────────────────┐       │                        │
│  │  │ ○───●───────────────────────│  0.18 │                        │
│  │  │ 0.00       0.25       0.50 │       │                        │
│  │  │                             │       │                        │
│  │  │  Current: 0.18 (Optimal)    │       │                        │
│  │  │  Custom:  0.22              │       │                        │
│  │  └─────────────────────────────┘       │                        │
│  │                                         │                        │
│  │  ECONOMIC SCENARIO                      │                        │
│  │  ┌────────────────────────────────┐     │                        │
│  │  │  ○ Base Case (Current)         │     │                        │
│  │  │  ○ Mild Recession (+20% PD)    │     │                        │
│  │  │  ○ Severe Recession (+50% PD)  │     │                        │
│  │  │  ○ Boom (−15% PD)              │     │                        │
│  │  └────────────────────────────────┘     │                        │
│  └────────────────────────────────────────┘                        │
├─────────────────────────────────────────────────────────────────────┤
│  KPI ROW (Updates in Real Time)                                     │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────────────┐   │
│  │ Approval │ Default  │ Net      │ Loss     │ Risk-Adj Return  │   │
│  │ Rate     │ Rate     │ Profit   │ Rate     │                  │   │
│  │ 83.5%    │ 3.5%     │ $71.1M   │ 3.0%     │ 13.5%            │   │
│  │ vs 81.0% │ vs 2.8%  │ vs $72.3M│ vs 2.4%  │ vs 14.3%         │   │
│  │ (δ +2.5) │ (δ +0.7) │ (δ −1.2M)│ (δ +0.6) │ (δ −0.8)         │   │
│  └──────────┴──────────┴──────────┴──────────┴──────────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│  CHART: Profit Curve (Line) with Current Setting Marked             │
│  ┌─────────────────────────────────────────────┐                    │
│  │  $75M ┌───────────────────┐                 │                    │
│  │       │              ★    │  Optimal         │                    │
│  │  $70M │        ★          │  ★ Current Sim   │                    │
│  │       │  ★                │                 │                    │
│  │  $60M ★───────────────────┘                 │                    │
│  │      0.05  0.10  0.15  0.20  0.25           │                    │
│  └─────────────────────────────────────────────┘                    │
├─────────────────────────────────────────────────────────────────────┤
│  TABLE: Scenario Comparison (Static Reference)                      │
│  ┌────────────┬────────┬────────┬────────┬────────┬──────────────┐  │
│  │ Scenario   │ Approv │ Default│ Profit │ Loss   │ Return       │  │
│  ├────────────┼────────┼────────┼────────┼────────┼──────────────┤  │
│  │ Approve All│ 100%   │ 8.0%   │ $56.1M │ 6.79%  │ 8.97%        │  │
│  │ Current    │ 76.6%  │ 3.7%   │ $64.0M │ 3.14%  │ 13.36%       │  │
│  │ AI Optimal │ 81.0%  │ 2.8%   │ $72.3M │ 2.38%  │ 14.28%       │  │
│  │ Custom     │ 83.5%  │ 3.5%   │ $71.1M │ 3.00%  │ 13.50%       │  │
│  └────────────┴────────┴────────┴────────┴────────┴──────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### KPIs (All React to Slider Changes)

| KPI | Base (Optimal) | Your Scenario | Delta |
|---|---|---|---|
| Approval Rate | 81.0% | 83.5% | +2.5 pp |
| Default Rate | 2.8% | 3.5% | +0.7 pp |
| Net Profit | $72.3M | $71.1M | −$1.2M |
| Expected Loss Rate | 2.38% | 3.00% | +0.62 pp |
| Risk-Adjusted Return | 14.28% | 13.50% | −0.78 pp |
| Loans Funded | 40,500 | 41,750 | +1,250 |

### Charts

| Chart | Type | Interactivity |
|---|---|---|
| Profit Curve | Line with marker | Threshold slider updates marker position |
| Scenario Comparison Table | Matrix | Highlights the selected scenario row |
| Delta Arrow Set | Custom KPI | Green/Red arrows for improvement/decline |

### Page-Level Controls

- **Threshold slider:** Core interaction. Drag to change the PD cutoff.
- **Economic scenario radio buttons:** Base / Mild Recession / Severe Recession / Boom (multiplies PDs by a scenario factor).
- **"Reset to Optimal" button:** Bookmark that resets threshold to 0.18.

---

## Page 5: Collections & Fairness Monitor (Head of Collections / Compliance)

**Question answered:** *"Which accounts should we call first? Is the model treating all groups fairly?"*

### Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│  HEADER: "Collections Prioritization & Fairness Monitor"            │
├────────────────────────────┬────────────────────────────────────────┤
│  COLLECTIONS PANEL          │  FAIRNESS PANEL                       │
│  ┌──────────────────────┐   │  ┌────────────────────────────────┐   │
│  │  KPI: Defaults       │   │  │  CHART: Approval Rate by       │   │
│  │  Caught in Top 5%    │   │  │  Age Group (Column Chart)      │   │
│  │  40% (454 / 1,134)   │   │  │  ┌────────────────────────┐    │   │
│  │                      │   │  │  │ <25  ██████████ 75%    │    │   │
│  │  KPI: LGD Reduction  │   │  │  │ 25-35████████████ 82%  │    │   │
│  │  20% (60% → 48%)     │   │  │  │ 35-50██████████████ 85%│    │   │
│  │                      │   │  │  │ 50+  ██████████████ 84%│    │   │
│  │  KPI: Net Savings    │   │  │  └────────────────────────┘    │   │
│  │  $529K                │   │  │                               │   │
│  │                      │   │  │  CHART: Disparate Impact      │   │
│  │  TABLE: Top 20       │   │  │  Ratio (Gauge Chart)          │   │
│  │  Accounts to Call    │   │  │  ┌────────────────┐           │   │
│  │  ┌──────┬─────┬────┐ │   │  │  │    0.92        │           │   │
│  │  │ Rank │ ID  │ PD │ │   │  │  └──┬───┬────┬────┤           │   │
│  │  │ 1    │ 4521│0.42│ │   │  │    0.8 0.9 1.0 1.1           │   │
│  │  │ 2    │ 8832│0.39│ │   │  │   Red Ambr Green              │   │
│  │  │ 3    │ 1204│0.37│ │   │  │                               │   │
│  │  │ ...  │ ... │ ...│ │   │  │  TABLE: Fairness Metrics      │   │
│  │  └──────┴─────┴────┘ │   │  │  ┌────────┬──────┬──────┬───┐ │   │
│  └──────────────────────┘   │  │  │ Group  │ App  │ Def  │DIR│ │   │
│                              │  │  ├────────┼──────┼──────┼───┤ │   │
│                              │  │  │ Age<25 │ 75%  │ 12%  │0.88│ │   │
│                              │  │  │ Age50+ │ 84%  │ 2%   │0.99│ │   │
│                              │  │  │ Employ │ 83%  │ 3%   │—   │ │   │
│                              │  │  │ Unempl │ 55%  │ 18%  │0.66│ │   │
│                              │  │  └────────┴──────┴──────┴───┘ │   │
│                              │  └────────────────────────────────┘   │
├────────────────────────────┴────────────────────────────────────────┤
│  ALERT BAR: Conditional formatting for fairness breaches             │
│  ⚠ Amber: Age < 25 approval rate is 75% vs. 84% for 50+ (DIR 0.89) │
│  🟢 Green: No gender-based disparity detected.                       │
│  🔴 RED: Unemployed applicants show DIR 0.66 — below 0.80 threshold │
└─────────────────────────────────────────────────────────────────────┘
```

### KPIs (Collections)

| KPI | Value |
|---|---|
| Defaults Caught in Top 5% | 40% (454 of 1,134 defaults) |
| LGD Reduction from Intervention | 20% (60% → 48%) |
| Net Savings from AI Collections | $529,125 |
| Accounts Flagged for Review | 2,025 (top 5% of 40,500) |

### KPIs (Fairness)

| KPI | Value | Status |
|---|---|---|
| Disparate Impact Ratio (Age < 25) | 0.89 | 🟡 Amber (monitor) |
| Disparate Impact Ratio (Unemployed) | 0.66 | 🔴 Red (escalate) |
| Disparate Impact Ratio (Age 50+) | 0.99 | 🟢 Green |
| Groups Below 0.80 Threshold | 1 (Unemployed) | 🔴 Action required |

### Charts

| Chart | Type | Purpose |
|---|---|---|
| Top 20 Collections Priority | Table | Actionable list for collections team |
| Approval Rate by Age Group | Column | Fairness visualisation |
| Disparate Impact Ratio Gauge | Radial Gauge | Quick status check (target > 0.80) |
| Fairness Metrics Table | Matrix | Full detail with DIR per group |
| Alert Bar | Custom visual | Traffic-light status per fairness dimension |

### Page-Level Filters

- **Collections:** Segment dropdown (All / Thin File / Overextended / etc.)
- **Fairness:** Group selector (Age / Income / Employment)

---

## Color Palette & Styling

| Element | Color | Hex |
|---|---|---|
| Primary (Horizon Blue) | Dark blue | #003366 |
| Secondary (Risk Red) | Red | #CC3333 |
| Tertiary (Safe Green) | Green | #339933 |
| Accent (Warning Amber) | Amber | #FF9933 |
| Background | Light grey | #F5F5F5 |
| Grid / Border | Medium grey | #CCCCCC |
| Text (Primary) | Dark grey | #333333 |
| Text (KPI Value) | Horizon Blue | #003366 |
| Positive Delta | Safe Green | #339933 |
| Negative Delta | Risk Red | #CC3333 |

### Conditional Formatting Rules

| Metric | Green | Amber | Red |
|---|---|---|---|
| Default Rate | < 3% | 3–5% | > 5% |
| Approval Rate | > 80% | 70–80% | < 70% |
| Risk-Adj Return | > 13% | 10–13% | < 10% |
| Disparate Impact Ratio | ≥ 0.95 | 0.80–0.95 | < 0.80 |
| Net Profit Trend | Increasing | Flat | Decreasing |

---

## Data Refresh & Automation

| Table | Refresh Frequency | Method |
|---|---|---|
| `applications` | Daily (midnight) | Python ETL script → CSV export → Power BI import |
| `features` | Daily | Same ETL script |
| `thresholds` | Weekly (or after model retraining) | Python simulation script |
| `collections` | Weekly | Python simulation script |
| `fairness` | Monthly | Python fairness audit script |

### Deployment Notes

1. **Dataset mode:** Import mode (daily refresh). DirectQuery is not needed for this volume (~50K rows).
2. **Row-level security (RLS):** Not required for this design but can be added if different managers should see different portfolio segments.
3. **Mobile layout:** Create a simplified mobile version with only Page 1 KPIs and the threshold slider.
4. **Export:** All pages support PDF export for board meeting packets.

---

## Implementation Order (Recommended Build Sequence)

1. **Page 1 (Executive Summary)** — Establish the core KPIs and threshold slider. This is the most-used page and should be built first.
2. **Page 4 (What-If Simulator)** — Build the interactive threshold slider logic. This leverages the same measures as Page 1.
3. **Page 2 (Risk Segmentation)** — Add dimension-based analysis using the `applications` table.
4. **Page 3 (Explainability)** — Import pre-computed SHAP values from the Python notebook. This is the most complex visual.
5. **Page 5 (Collections & Fairness)** — Add the collections ranking table and fairness metrics. This depends on fairness audit outputs.

---

*End of Dashboard Design Document.*
