# Model Evaluation Design — Design Document

> Extracted from `src/evaluate.py` (architecture specification)

## Overview
evaluate.py — Model Evaluation & Threshold Optimization for Credit Risk Analytics

This module provides comprehensive evaluation of trained credit risk models.
It is designed to be called from notebooks via `from src.evaluate import ...`
and operates on model predictions from the test set.

Key responsibilities:
    1. Compute all standard classification metrics (AUC, precision, recall, F1, etc.)
    2. Tune the probability decision threshold to maximize business utility
    3. Assess calibration (reliability of predicted probabilities)
    4. Generate evaluation plots (ROC, PR, calibration, KS, confusion matrix)
    5. Compute Gini coefficient and KS statistic (industry-standard for credit risk)
    6. Produce a model card summary for compliance documentation

Usage:
    from src.evaluate import ModelEvaluator

    evaluator = ModelEvaluator()
    results = evaluator.evaluate(y_true, y_pred_proba)

    # Threshold tuning for business
    best_threshold = evaluator.tune_threshold(y_true, y_pred_proba, metric='profit')

    # Calibration assessment
    calibration_report = evaluator.calibrate_if_needed(y_true, y_pred_proba)

    # Full report
    evaluator.summary_report(y_true, y_pred_proba, save_path='reports/model_card.md')

Author: [Portfolio Project]
Date: July 2026

## Pipeline Classes
- **MetricsCalculator**: Compute all standard and credit-risk-specific metrics.
- **ThresholdTuner**: Find the optimal probability threshold for credit decision making.
- **CalibrationAssessor**: Assess and optionally correct probability calibration.
- **EvaluationVisualizer**: Generate all evaluation plots for reporting and presentation.
- **ModelCardGenerator**: Generate a model card — a concise summary of model performance,
- **ModelComparer**: Compare multiple trained models on the same test set.
- **ModelEvaluator**: Top-level class that orchestrates all evaluation activities.

## Utility Functions
- `binarize_predictions()` — Convert predicted probabilities to binary predictions using a threshold.
- `compute_profit_curve()` — Compute expected profit at every possible threshold.
- `find_optimal_threshold()` — Generic threshold optimizer.
- `format_metrics_table()` — Convert metrics dictionary to a formatted markdown table.
- `save_evaluation_report()` — Save evaluation results as a JSON file for reproducibility.