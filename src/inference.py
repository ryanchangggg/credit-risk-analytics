"""
inference.py — Model Inference and Scoring for Credit Risk Analytics

This module loads the trained LightGBM model and provides a clean API for
scoring new loan applicants. It handles model loading, feature validation,
probability scoring, and result formatting.

Typical usage:
    from src.inference import CreditRiskScorer

    scorer = CreditRiskScorer()
    scorer.load_model("models/best_model.pkl")

    # Score pre-processed applicant data
    predictions = scorer.predict(applicant_df)

    # Get detailed risk assessment
    assessment = scorer.assess_risk(applicant_df)
    print(assessment[["SK_ID_CURR", "pd_score", "risk_bucket", "decision"]].head())

Author: [Portfolio Project]
Date: July 2026
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

logger = logging.getLogger(__name__)

# Risk bucket definitions (matches business simulation risk segmentation)
RISK_BUCKETS = [
    ("A - Lowest Risk", 0.00, 0.02, "Auto-approve; offer preferred rates"),
    ("B - Low Risk", 0.02, 0.05, "Auto-approve; standard terms"),
    ("C - Medium Risk", 0.05, 0.10, "Review; standard terms"),
    ("D - High Risk", 0.10, 0.30, "Enhanced review; higher APR or reduced amount"),
    ("E - Highest Risk", 0.30, 1.00, "Likely decline; refer to manual underwriting"),
]

DEFAULT_THRESHOLD = 0.485
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.pkl"


class CreditRiskScorer:
    """Loads a trained LightGBM model and scores loan applicants.

    Parameters
    ----------
    model_path : str or Path, optional
        Path to the saved model dict (``best_model.pkl``).  If not given,
        you must call ``load_model()`` before scoring.
    threshold : float, optional
        Decision threshold for binary classification (default 0.485).
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        self.model: Optional[object] = None
        self.model_name: Optional[str] = None
        self.feature_names: Optional[List[str]] = None
        self.threshold = threshold
        self._loaded = False

        if model_path is not None:
            self.load_model(model_path)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self, model_path: Union[str, Path]) -> Dict:
        """Load a trained model from disk.

        Parameters
        ----------
        model_path : str or Path
            Path to the ``best_model.pkl`` bundle saved by the training pipeline.

        Returns
        -------
        dict
            The raw model bundle (useful for inspection).
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        import joblib

        bundle = joblib.load(str(model_path))

        self.model = bundle["model"]
        self.model_name = bundle.get("model_name", "Unknown")
        self.feature_names = bundle.get("feature_names", [])
        self._loaded = True

        logger.info(
            "Loaded %s model (%d features) from %s",
            self.model_name,
            len(self.feature_names),
            model_path,
        )
        return bundle

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def _validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure the input DataFrame has the expected feature columns.

        * Warns about and drops unknown columns.
        * Raises ``ValueError`` if required columns are missing.
        * Returns a DataFrame with columns ordered to match the model.
        """
        if not self._loaded:
            raise RuntimeError("No model loaded. Call load_model() first.")

        required = set(self.feature_names)
        provided = set(df.columns)

        # Unknown columns (not used by the model)
        extra = provided - required
        if extra:
            logger.warning("Dropping %d unknown column(s): %s", len(extra), sorted(extra)[:5])

        # Missing columns
        missing = required - provided
        if missing:
            raise ValueError(
                f"Missing %d required feature(s): %s" % (len(missing), sorted(missing)[:10])
            )

        # Align and order
        return df[[col for col in self.feature_names if col in df.columns]]

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return predicted default probabilities for each applicant.

        Parameters
        ----------
        df : pd.DataFrame
            Feature matrix with columns matching the model's ``feature_names``.

        Returns
        -------
        np.ndarray
            Shape ``(n_applicants,)`` — probability of default (``TARGET=1``).
        """
        X = self._validate(df)
        proba = self.model.predict_proba(X)
        return proba[:, 1]  # probability of class 1 (default)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return binary default predictions using the configured threshold.

        Parameters
        ----------
        df : pd.DataFrame
            Feature matrix with columns matching the model's ``feature_names``.

        Returns
        -------
        np.ndarray
            Shape ``(n_applicants,)`` — ``1`` if predicted to default, else ``0``.
        """
        proba = self.predict_proba(df)
        return (proba >= self.threshold).astype(int)

    # ------------------------------------------------------------------
    # Risk assessment
    # ------------------------------------------------------------------

    def assess_risk(
        self,
        df: pd.DataFrame,
        loan_amount_col: Optional[str] = "AMT_CREDIT",
        id_col: Optional[str] = "SK_ID_CURR",
    ) -> pd.DataFrame:
        """Score applicants and return a comprehensive risk assessment.

        The returned DataFrame contains probability scores, risk buckets,
        binary decisions, and optional loan-amount information.

        Parameters
        ----------
        df : pd.DataFrame
            Feature matrix.
        loan_amount_col : str or None
            Column name for loan amount (used for expected-loss calculation).
            Pass ``None`` to skip.
        id_col : str or None
            Column name for applicant ID.  Pass ``None`` to skip.

        Returns
        -------
        pd.DataFrame
            Columns: ``SK_ID_CURR``, ``pd_score``, ``risk_bucket``, ``bucket_desc``,
            ``predicted_default``, ``decision``, (optional) ``expected_loss``.
        """
        proba = self.predict_proba(df)
        binary = (proba >= self.threshold).astype(int)

        # Assign risk buckets
        bucket_labels = []
        bucket_descs = []
        for p in proba:
            for label, lo, hi, desc in RISK_BUCKETS:
                if lo <= p < hi:
                    bucket_labels.append(label)
                    bucket_descs.append(desc)
                    break
            else:
                bucket_labels.append("E - Highest Risk")
                bucket_descs.append("Likely decline; refer to manual underwriting")

        result = pd.DataFrame(
            {
                "pd_score": np.round(proba, 4),
                "risk_bucket": bucket_labels,
                "bucket_desc": bucket_descs,
                "predicted_default": binary,
                "decision": np.where(
                    binary == 1, "Decline / Enhanced Review", "Approve"
                ),
            },
            index=df.index,
        )

        if id_col and id_col in df.columns:
            result.insert(0, id_col, df[id_col].values)

        if loan_amount_col and loan_amount_col in df.columns:
            result["expected_loss"] = np.round(
                proba * df[loan_amount_col].values * 0.60, 2
            )  # LGD = 60%

        return result

    # ------------------------------------------------------------------
    # Batch scoring
    # ------------------------------------------------------------------

    def score_csv(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        id_col: str = "SK_ID_CURR",
        loan_amount_col: str = "AMT_CREDIT",
    ) -> pd.DataFrame:
        """Score applicants from a CSV file and save the results.

        Parameters
        ----------
        input_path : str or Path
            Path to input CSV (must contain the required feature columns).
        output_path : str or Path
            Path for the output CSV with scores and risk assessments.
        id_col : str
            Applicant ID column name.
        loan_amount_col : str
            Loan amount column for expected-loss calculation.

        Returns
        -------
        pd.DataFrame
            The scored result (also saved to ``output_path``).
        """
        df = pd.read_csv(input_path)
        result = self.assess_risk(df, loan_amount_col=loan_amount_col, id_col=id_col)
        result.to_csv(output_path, index=False)
        logger.info("Scored %d applicants -> %s", len(result), output_path)
        return result


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def main():
    """Simple CLI: score a CSV file.

    Usage:
        python src/inference.py data/processed/sample.csv --output scored.csv
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Score loan applicants with the trained credit risk model."
    )
    parser.add_argument(
        "input_csv",
        type=str,
        help="Path to input CSV with feature columns.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="scored_applicants.csv",
        help="Output CSV path (default: scored_applicants.csv).",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=str(MODEL_PATH),
        help=f"Model path (default: {MODEL_PATH}).",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Decision threshold (default: {DEFAULT_THRESHOLD}).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    scorer = CreditRiskScorer(model_path=args.model, threshold=args.threshold)
    result = scorer.score_csv(args.input_csv, args.output)
    accepted = (result["predicted_default"] == 0).sum()
    declined = (result["predicted_default"] == 1).sum()
    print(f"Decision summary: {accepted} accepted, {declined} declined")
    print(f"Average PD of accepted: {result.loc[result.predicted_default == 0, 'pd_score'].mean():.4f}")


if __name__ == "__main__":
    main()
