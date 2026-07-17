# Feature Engineering Pipeline — Design Document

> Extracted from `src/preprocess.py` (architecture specification)

## Overview
preprocess.py — Feature Engineering Pipeline for Credit Risk Analytics

This module implements the complete feature engineering design documented in
notebooks/02_feature_engineering.ipynb. It transforms raw Home Credit data
tables into a master feature matrix ready for modeling.

Pipeline overview (14 steps):
    1. Load and validate all 8 raw tables
    2. Clean application_train (artifact fix, outlier capping)
    3. Create domain ratio features (Group A)
    4. Aggregate bureau table (Group C)
    5. Aggregate bureau_balance (Group C8)
    6. Aggregate previous_application (Group D)
    7. Aggregate installments_payments (Group E)
    8. Aggregate POS_CASH_balance (Group F)
    9. Aggregate credit_card_balance (Group G)
    10. Merge all aggregations into master DataFrame
    11. Encode categorical features (Group H)
    12. Create interaction features (Group J)
    13. Create missing indicators and impute (Group I)
    14. Feature selection (Group K)

Usage:
    from src.preprocess import build_feature_pipeline

    pipeline = build_feature_pipeline(config)
    X_train = pipeline.fit_transform(train_tables)
    X_test = pipeline.transform(test_tables)

Author: [Portfolio Project]
Date: July 2026

## Pipeline Steps
- **DaysEmployedCleaner**: Fix the DAYS_EMPLOYED artifact (365243 = unemployed placeholder).
- **OutlierCapper**: Cap extreme values at specified percentiles.
- **DomainRatioFeatures**: Create domain-specific ratio features from application_train columns.
- **BureauAggregator**: Aggregate the bureau table into one row per applicant.
- **BureauBalanceAggregator**: Aggregate bureau_balance table and merge into bureau aggregates.
- **PreviousApplicationAggregator**: Aggregate previous_application table into one row per applicant.
- **InstallmentAggregator**: Aggregate installments_payments into one row per applicant.
- **PosCashAggregator**: Aggregate POS_CASH_balance into one row per applicant.
- **CreditCardAggregator**: Aggregate credit_card_balance into one row per applicant.
- **MasterMerger**: Merge all aggregated tables with application_train on SK_ID_CURR.
- **CategoricalEncoder**: Apply mixed encoding strategy to categorical columns.
- **InteractionFeatures**: Create interaction features from selected pairs of base features.
- **MissingValueHandler**: Create missing-indicator flags and impute remaining NaN values.
- **FeatureSelector**: Reduce feature dimensionality using a multi-stage selection process.
- **FeatureScaler**: Apply RobustScaler to numeric features (handles outliers better than StandardScaler).

## Utility Functions
- `load_raw_data()` — Load Home Credit raw CSV files from the specified directory.
- `validate_primary_keys()` — Verify that primary keys are unique and consistent across tables.
- `build_feature_pipeline()` — Build the complete feature engineering pipeline.
- `log_transform()` — Apply log1p transformation for right-skewed features.
- `create_dti()` — Compute Debt-to-Income Ratio with safety checks.
- `create_lti()` — Compute Loan-to-Income Ratio with safety checks.
- `age_from_days()` — Convert DAYS_BIRTH (negative days) to age in years.
- `binary_flag()` — Create a binary indicator (0/1) based on a condition.
- `missing_rate()` — Compute the fraction of missing values in a series.
- `get_feature_names()` — Extract feature names from sklearn transformers that support get_feature_names_out.