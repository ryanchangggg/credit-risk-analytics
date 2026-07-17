# Model Training Pipeline — Design Document

> Extracted from `src/train.py` (architecture specification)

## Overview
train.py — Model Training Pipeline for Credit Risk Analytics

This module handles model training, hyperparameter tuning, cross-validation,
and model persistence. It is designed to be called from notebooks via
`from src.train import ...` and to operate on the feature matrix produced
by `src.preprocess.build_feature_pipeline()`.

Pipeline overview:
    1. Load preprocessed feature matrix (X, y) from parquet / CSV
    2. Perform stratified train/validation/test split
    3. Train multiple candidate models with reasonable default hyperparameters
    4. Tune top-2 models via Optuna Bayesian optimization
    5. Select best model based on validation AUC and business metrics
    6. Retrain on train + validation and evaluate on held-out test set
    7. Persist trained model, scaler, and feature names to disk

Usage:
    from src.preprocess import build_feature_pipeline
    from src.train import ModelTrainer, train_and_select

    pipeline = build_feature_pipeline()
    X_train = pipeline.fit_transform(train_tables)
    X_test = pipeline.transform(test_tables)

    trainer = ModelTrainer(config)
    results = trainer.run(X_train, y_train)

    best_model = results['best_model']
    best_params = results['best_params']
    cv_scores = results['cv_scores']
    leaderboard = results['leaderboard']

Author: [Portfolio Project]
Date: July 2026

## Pipeline Classes
- **DataSplitter**: Perform stratified train/validation/test split.
- **CrossValidator**: Perform stratified k-fold cross-validation with multiple scoring metrics.
- **LogisticRegressionModel**: Logistic Regression with L2 regularization and class weighting.
- **DecisionTreeModel**: Decision Tree (CART) with pruning to avoid overfitting.
- **RandomForestModel**: Random Forest ensemble of 200+ decision trees.
- **XGBoostModel**: XGBoost — Extreme Gradient Boosting.
- **LightGBMModel**: LightGBM — Light Gradient Boosting Machine.
- **HyperparameterTuner**: Bayesian hyperparameter optimization using Optuna.
- **ModelLeaderboard**: Compare all candidate models on a consistent set of metrics.
- **ModelPersister**: Save and load trained models, scalers, and metadata to/from disk.
- **ModelTrainer**: Orchestrate the full training pipeline.

## Utility Functions
- `prepare_data()` — Validate and prepare data for modeling.
- `compute_scale_pos_weight()` — Compute the scale_pos_weight for XGBoost / LightGBM.
- `get_default_params()` — Return default hyperparameters for a given model type.