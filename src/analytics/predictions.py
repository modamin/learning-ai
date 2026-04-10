"""
Sprint 2 — Task 2.3: Local dropout prediction fallback.

Trains the same GradientBoostingClassifier used in the Fabric notebook,
but runs locally against data fetched from the Fabric SQL analytics endpoint.
The trained model is cached in-memory so subsequent calls are instant.

Usage
-----
from src.analytics.predictions import DropoutPredictor
predictor = DropoutPredictor.from_analytics(analytics)   # trains on first call
risk = predictor.predict_dropout_risk({"total_events": 15, "modules_touched": 2, ...})
"""
from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from src.analytics.queries import FabricAnalytics

logger = logging.getLogger(__name__)

# Features must match exactly what the Fabric notebook trains on
FEATURE_COLS = [
    "total_events",
    "modules_touched",
    "avg_score",
    "active_days",
    "span_days",
    "events_per_day",
    "completion_velocity",
    "score_x_events",
]

GBM_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_leaf=20,
    random_state=42,
)


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features — same logic as the Fabric notebook."""
    df = df.copy()
    df["avg_score"] = df["avg_score"].fillna(df["avg_score"].median())
    df["events_per_day"] = df["total_events"] / df["active_days"].clip(lower=1)
    df["completion_velocity"] = df["modules_touched"] / df["span_days"].clip(lower=1)
    df["score_x_events"] = df["avg_score"] * df["total_events"]
    return df


class DropoutPredictor:
    """
    Locally-trained dropout risk model backed by Fabric SQL data.

    Thread-safe: training is protected by a lock so concurrent Streamlit
    sessions don't trigger duplicate training.
    """

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self._feature_importances: dict[str, float] = {}
        self._train_roc_auc: float = 0.0
        self._n_training_samples: int = 0
        self._lock = threading.Lock()

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_analytics(cls, analytics: "FabricAnalytics") -> "DropoutPredictor":
        """Fetch training data from Fabric SQL and train the model."""
        predictor = cls()
        predictor.train(analytics)
        return predictor

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, analytics: "FabricAnalytics") -> None:
        """
        Fetch feature data from Fabric SQL and train the GBM pipeline.
        Safe to call multiple times — re-trains and replaces the cached model.
        """
        with self._lock:
            logger.info("Fetching training data from Fabric SQL ...")
            df_raw = analytics.get_training_features()

            if df_raw.empty:
                raise ValueError(
                    "No training data returned. Ensure silver_xapi_events is populated."
                )

            df = _engineer_features(df_raw)
            X = df[FEATURE_COLS].fillna(0)
            y = df["completed"].astype(int)

            self._n_training_samples = len(X)
            logger.info("Training on %d learner-course pairs ...", self._n_training_samples)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.20, random_state=42, stratify=y
            )

            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("gbm", GradientBoostingClassifier(**GBM_PARAMS)),
            ])
            pipeline.fit(X_train, y_train)

            y_proba = pipeline.predict_proba(X_test)[:, 1]
            self._train_roc_auc = roc_auc_score(y_test, y_proba)
            self._feature_importances = dict(
                zip(FEATURE_COLS, pipeline.named_steps["gbm"].feature_importances_)
            )
            self._pipeline = pipeline

            logger.info(
                "Model trained. ROC-AUC=%.3f  n=%d",
                self._train_roc_auc,
                self._n_training_samples,
            )

    # ── Inference ─────────────────────────────────────────────────────────────

    def _check_trained(self) -> None:
        if self._pipeline is None:
            raise RuntimeError(
                "Model not trained. Call .train(analytics) or use .from_analytics()."
            )

    def predict_dropout_risk(self, learner_features: dict) -> float:
        """
        Return the probability that a learner will NOT complete a course (0–1).

        Parameters
        ----------
        learner_features : dict
            Must contain at least the raw keys:
            total_events, modules_touched, avg_score, active_days, span_days
            Derived features (events_per_day, etc.) are computed internally.

        Returns
        -------
        float
            Dropout risk probability in [0, 1]. Higher = more likely to drop.
        """
        self._check_trained()
        row = pd.DataFrame([learner_features])
        row = _engineer_features(row)
        X = row[FEATURE_COLS].fillna(0)
        # predict_proba[:, 0] = P(NOT completed) = dropout risk
        risk: float = self._pipeline.predict_proba(X)[0, 0]
        return round(float(risk), 4)

    def predict_batch(self, df: pd.DataFrame) -> pd.Series:
        """
        Score a DataFrame of learner features.

        Parameters
        ----------
        df : pd.DataFrame
            Rows are learner-course pairs. Must contain the base feature columns.

        Returns
        -------
        pd.Series
            Dropout risk scores (0–1), same index as df.
        """
        self._check_trained()
        df_feat = _engineer_features(df)
        X = df_feat[FEATURE_COLS].fillna(0)
        proba = self._pipeline.predict_proba(X)[:, 0]
        return pd.Series(proba, index=df.index, name="dropout_risk")

    def predict_for_course(
        self,
        analytics: "FabricAnalytics",
        course_id: str,
    ) -> pd.DataFrame:
        """
        Fetch all active learners for a course from Fabric SQL, score them,
        and return a DataFrame sorted by dropout risk (highest first).

        Returned columns:
          learner_email, modules_touched, avg_score, active_days,
          span_days, dropout_risk, risk_tier
        """
        self._check_trained()

        sql = """
            SELECT
                learner_email,
                COUNT(*)                              AS total_events,
                COUNT(DISTINCT module_id)             AS modules_touched,
                ROUND(AVG(CASE WHEN verb = 'scored' THEN score END), 4) AS avg_score,
                COUNT(DISTINCT CAST(ts AS DATE))      AS active_days,
                DATEDIFF(day, MIN(ts), MAX(ts))       AS span_days
            FROM silver_xapi_events
            WHERE course_id = ?
              AND learner_email NOT IN (
                  SELECT learner_email
                  FROM silver_xapi_events
                  WHERE course_id = ?
                    AND verb = 'completed'
                    AND course_completed = 1
              )
            GROUP BY learner_email
            HAVING COUNT(*) >= 3
        """
        df = analytics._df(sql, [course_id, course_id])

        if df.empty:
            return df

        df["dropout_risk"] = self.predict_batch(df)
        df["risk_tier"] = pd.cut(
            df["dropout_risk"],
            bins=[0, 0.40, 0.65, 1.0],
            labels=["Low", "Medium", "High"],
        )
        return df.sort_values("dropout_risk", ascending=False).reset_index(drop=True)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    @property
    def is_trained(self) -> bool:
        return self._pipeline is not None

    @property
    def model_info(self) -> dict:
        """Return a summary dict for display in the UI."""
        return {
            "trained": self.is_trained,
            "roc_auc": round(self._train_roc_auc, 4),
            "n_samples": self._n_training_samples,
            "feature_importances": {
                k: round(v, 4)
                for k, v in sorted(
                    self._feature_importances.items(), key=lambda x: -x[1]
                )
            },
        }

    def risk_label(self, score: float) -> str:
        """Convert a raw dropout risk score to a human-readable label."""
        if score >= 0.65:
            return "High"
        if score >= 0.40:
            return "Medium"
        return "Low"

    def suggested_intervention(self, score: float, days_inactive: int = 0) -> str:
        """Return a suggested L&D intervention based on risk tier."""
        if score >= 0.65:
            if days_inactive > 14:
                return "Send re-engagement email + offer 1:1 coaching session"
            return "Proactive check-in from manager; highlight upcoming module value"
        if score >= 0.40:
            if days_inactive > 7:
                return "Automated reminder + link to module summary"
            return "Peer study group recommendation"
        return "On track — no intervention needed"
