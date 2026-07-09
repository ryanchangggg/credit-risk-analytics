"""
Unit tests for the inference module (src/inference.py).

These tests verify model loading, feature validation, scoring, and
risk assessment logic using the actual trained model.
"""

import sys
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import CreditRiskScorer, RISK_BUCKETS, DEFAULT_THRESHOLD

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
HAS_MODEL = MODEL_PATH.exists()


@pytest.fixture(scope="session")
def scorer():
    if not HAS_MODEL:
        pytest.skip("Model file not found -- run model training first")
    return CreditRiskScorer(model_path=str(MODEL_PATH))


@pytest.fixture
def sample_features(scorer):
    np.random.seed(42)
    n = 10
    data = {col: np.random.randn(n) for col in scorer.feature_names}
    data["SK_ID_CURR"] = range(10001, 10001 + n)
    return pd.DataFrame(data)


class TestModelLoading:
    def test_load_model(self):
        if not HAS_MODEL:
            pytest.skip("Model file not found")
        s = CreditRiskScorer(str(MODEL_PATH))
        assert s._loaded
        assert s.model is not None
        assert s.model_name == "LightGBM"
        assert len(s.feature_names) == 182

    def test_load_model_missing_file(self):
        with pytest.raises(FileNotFoundError):
            CreditRiskScorer("nonexistent.pkl")

    def test_predict_before_load_raises(self):
        s = CreditRiskScorer()
        with pytest.raises(RuntimeError, match="No model loaded"):
            s.predict(pd.DataFrame())


class TestFeatureValidation:
    def test_correct_features_pass(self, scorer, sample_features):
        result = scorer.predict(sample_features)
        assert len(result) == len(sample_features)

    def test_extra_columns_dropped(self, scorer, sample_features):
        sample_features["EXTRA_COL"] = 999
        result = scorer.predict(sample_features)
        assert len(result) == len(sample_features)

    def test_missing_column_raises(self, scorer, sample_features):
        bad = sample_features.drop(columns=["EXT_MEAN"])
        with pytest.raises(ValueError, match="Missing"):
            scorer.predict(bad)


class TestScoring:
    def test_predict_proba_returns_probabilities(self, scorer, sample_features):
        proba = scorer.predict_proba(sample_features)
        assert proba.shape == (len(sample_features),)
        assert proba.dtype == np.float64
        assert ((0 <= proba) & (proba <= 1)).all()

    def test_predict_returns_binary(self, scorer, sample_features):
        preds = scorer.predict(sample_features)
        assert preds.shape == (len(sample_features),)
        assert set(preds.tolist()).issubset({0, 1})

    def test_threshold_respected(self, scorer, sample_features):
        s = CreditRiskScorer(model_path=str(MODEL_PATH), threshold=0.0)
        preds = s.predict(sample_features)
        assert (preds == 1).all()

    def test_threshold_high(self, scorer, sample_features):
        s = CreditRiskScorer(model_path=str(MODEL_PATH), threshold=1.0)
        preds = s.predict(sample_features)
        assert (preds == 0).all()


class TestRiskAssessment:
    def test_assess_risk_columns(self, scorer, sample_features):
        result = scorer.assess_risk(sample_features)
        expected_cols = {"SK_ID_CURR", "pd_score", "risk_bucket", "bucket_desc",
                         "predicted_default", "decision", "expected_loss"}
        assert expected_cols.issubset(set(result.columns))

    def test_assess_risk_buckets_cover_range(self):
        boundaries = sorted(set([lo for _, lo, hi, _ in RISK_BUCKETS] +
                                [hi for _, lo, hi, _ in RISK_BUCKETS]))
        assert boundaries[0] == 0.0
        assert boundaries[-1] >= 0.99

    def test_bucket_assignment(self, scorer, sample_features):
        result = scorer.assess_risk(sample_features)
        for _, row in result.iterrows():
            pd_val = row["pd_score"]
            bucket = row["risk_bucket"]
            matched = any(
                lo <= pd_val < hi and label == bucket
                for label, lo, hi, _ in RISK_BUCKETS
            )
            assert matched, f"PD {pd_val} does not match bucket {bucket}"

    def test_expected_loss_calculation(self, scorer, sample_features):
        result = scorer.assess_risk(sample_features, loan_amount_col="AMT_CREDIT")
        expected = (result["pd_score"].values *
                    sample_features["AMT_CREDIT"].values * 0.60)
        assert np.allclose(result["expected_loss"].values, np.round(expected, 2), atol=0.01)


class TestBatchScoring:
    def test_score_csv(self, scorer, sample_features):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f_in:
            sample_features.to_csv(f_in, index=False)
            in_path = f_in.name
        out_path = in_path.replace(".csv", "_scored.csv")
        try:
            result = scorer.score_csv(in_path, out_path)
            assert os.path.exists(out_path)
            assert len(result) == len(sample_features)
            assert "pd_score" in result.columns
        finally:
            os.unlink(in_path)
            if os.path.exists(out_path):
                os.unlink(out_path)


class TestDefaults:
    def test_default_threshold(self):
        assert DEFAULT_THRESHOLD == 0.485

    def test_default_model_path_resolves(self):
        from src.inference import MODEL_PATH
        assert str(MODEL_PATH).endswith("models/best_model.pkl")
