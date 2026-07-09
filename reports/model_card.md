# Model Card — Credit Risk Default Prediction

## Model Summary

| Field | Value |
|---|---|
| **Model Type** | LightGBM (LGBMClassifier) |
| **Intended Use** | Predict probability of default (PD) for unsecured personal loan applicants |
| **Target** | Binary: 1 = defaulted (90+ DPD), 0 = repaid |
| **Base Rate** | 8.07% default |
| **Training Data** | 246,008 applicants (80% of 307,511) |
| **Validation** | Internal holdout: 61,503 applicants (20%) |

## Performance

| Metric | Value |
|---|---|
| **AUC-ROC** | 0.7680 |
| **Gini Coefficient** | 0.536 (2 x AUC - 1) |
| **Average Precision** | 0.258 |
| **Optimal Threshold (F2-optimized)** | 0.485 |
| **Precision at threshold** | 0.171 |
| **Recall at threshold** | 0.697 |
| **F1 Score** | 0.275 |
| **F2 Score** | 0.432 |

### Confusion Matrix (61,503 test applicants)

|  | Predicted Repaid | Predicted Default |
|---|---|---|
| **Actual Repaid** | 39,774 (TN) | 16,764 (FP) |
| **Actual Default** | 1,504 (FN) | 3,461 (TP) |

### Leaderboard

| Model | AUC | Avg Precision | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **LightGBM** | **0.768** | **0.258** | 0.176 | **0.676** | 0.279 |
| XGBoost | 0.767 | 0.256 | **0.181** | 0.651 | **0.283** |
| Logistic Regression | 0.748 | 0.236 | 0.159 | 0.677 | 0.258 |
| Random Forest | 0.746 | 0.227 | 0.183 | 0.590 | 0.279 |
| Decision Tree | 0.723 | 0.202 | 0.145 | 0.678 | 0.239 |

## Feature Engineering

- **182 features** engineered from applicant-level data
- Feature groups: external credit scores, financial ratios, demographics, interaction terms, frequency-encoded categoricals
- 62 missing-indicator flags created for high-missingness features

### Top-5 Features by SHAP Importance

| Rank | Feature | Mean SHAP | Business Interpretation |
|---|---|---|---|
| 1 | EXT_MEAN | 0.47 | Composite external creditworthiness (dominant driver) |
| 2 | ANNUITY_RATE | 0.17 | Debt burden: higher ratio = higher risk |
| 3 | AMT_GOODS_PRICE | 0.14 | Loan size signal: larger = more scrutiny needed |
| 4 | AMT_CREDIT | 0.07 | Credit amount relative to capacity |
| 5 | EXT_SOURCE_3 | 0.08 | Third-party credit score signal |

### Feature Processing

| Step | Method |
|---|---|
| Missing values | Median/mode imputation + 62 missing-indicator flags |
| Categorical encoding | Frequency encoding (high cardinality) + one-hot encoding |
| Outlier capping | Domain-informed (P0.1-P99.9) |
| Scaling | RobustScaler (median/IQR-based) |
| DAYS_EMPLOYED artifact | 365,243 days = unemployed flag created (18% of applicants) |

## Business Simulation

Simulated on a $625M portfolio (50,000 loans, $12,500 average principal):

| Policy | Approval Rate | Default Rate | Net Profit | Risk-Adj Return |
|---|---|---|---|---|
| Approve Everyone | 100.0% | 8.11% | $129.1M | 20.66% |
| LightGBM (threshold=0.485) | **67.1%** | **3.62%** | **$112.6M** | **26.83%** |
| Rule-Based (current) | 4.6% | 2.50% | $8.1M | 28.30% |

The AI model generates **$104.5M more profit** than the current rules-based policy (1,298% improvement), approving 67% of applicants vs. 4.6%.

## Risk Segmentation

| Bucket | PD Range | Strategy |
|---|---|---|
| A - Lowest Risk | < 2% | Auto-approve; preferred rates |
| B - Low Risk | 2-5% | Auto-approve; standard terms |
| C - Medium Risk | 5-10% | Standard review |
| D - High Risk | 10-30% | Enhanced review; adjusted terms |
| E - Highest Risk | > 30% | Likely decline; manual underwriting |

## Model Limitations

1. **AUC gap**: Achieved 0.768 vs. target 0.80. May underperform in portfolios with different base default rates.
2. **Single-table features**: Only the main `application_train` table was used. Performance may improve with multi-table aggregation (bureau, previous applications, installments).
3. **No temporal validation**: The train/test split is random, not time-based. Look-ahead bias is possible.
4. **Precision floor**: 17% precision at threshold is low; flagged applicants include many false positives, increasing collections cost.
5. **No fairness audit**: Disparate impact by protected attributes has not been formally assessed.
6. **Calibration**: Brier score and calibration curve have not been validated on the final model.

## Fairness Considerations

A formal fairness assessment has not been completed. Model features include `CODE_GENDER_M` and age-related variables, which may correlate with protected attributes. A disparate impact analysis should be conducted before production deployment.

## Intended Use

- **Primary**: Decision-support tool for loan underwriting at Horizon Lending Inc.
- **Secondary**: Collections prioritization on the funded portfolio
- **Not intended for**: Automated lending decisions without human review, use in portfolios with substantially different risk profiles, or use as the sole basis for adverse action notifications

## Technical Requirements

- Environment: Python 3.10+, LightGBM 4.0+, scikit-learn 1.3+
- Inference: See `src/inference.py` for the scoring API
- Training: See `execution/04_modeling.ipynb` for the full training pipeline
