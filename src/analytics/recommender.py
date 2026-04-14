"""
CourseRecommender — collaborative filtering via weighted implicit feedback + TruncatedSVD.

Complements the DropoutPredictor in predictions.py. Where the dropout model asks
"will this learner finish?", the recommender asks "which courses will this learner enjoy?"

Algorithm
---------
1. Build a sparse learner × course interaction matrix from silver_xapi_events.
   Interaction weight = log1p(event_count) + completed × 3 + (avg_score / 100)
2. Decompose with TruncatedSVD (k=50 latent factors).
3. At inference time, project a learner's row into latent space and rank all courses
   by dot-product similarity, filtering out courses the learner has already started.

Usage
-----
from src.analytics.recommender import CourseRecommender
rec = CourseRecommender.from_analytics(analytics)   # trains on first call
recs = rec.recommend("alice@example.com", top_n=5)  # [{"course_id": ..., "score": ...}]
"""
from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

if TYPE_CHECKING:
    from src.analytics.queries import FabricAnalytics

logger = logging.getLogger(__name__)

N_COMPONENTS = 50


class CourseRecommender:
    """
    SVD-based collaborative filtering recommender trained on xAPI event data.

    Thread-safe: training is protected by a lock so concurrent Streamlit
    sessions don't trigger duplicate training.
    """

    def __init__(self) -> None:
        self._svd: TruncatedSVD | None = None
        self._learner_factors: np.ndarray | None = None   # shape (n_learners, k)
        self._course_factors: np.ndarray | None = None    # shape (n_courses, k)
        self._learner_index: dict[str, int] = {}          # email → row index
        self._course_index: list[str] = []                # column index → course_id
        self._learner_courses: dict[str, set[str]] = {}   # email → started course_ids
        self._lock = threading.Lock()

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_analytics(cls, analytics: "FabricAnalytics") -> "CourseRecommender":
        """Fetch interaction data from Fabric SQL and train the model."""
        rec = cls()
        rec.train(analytics)
        return rec

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, analytics: "FabricAnalytics") -> None:
        """
        Fetch learner-course interaction data and fit the SVD factorisation.
        Safe to call multiple times — re-trains and replaces the cached model.
        """
        with self._lock:
            logger.info("Fetching interaction matrix data from Fabric SQL ...")
            df = analytics.get_interaction_matrix_data()

            if df.empty:
                raise ValueError(
                    "No interaction data returned. Ensure silver_xapi_events is populated."
                )

            # Interaction weight: engagement depth + completion bonus + quiz signal
            df["weight"] = (
                np.log1p(df["event_count"]) * 1.0
                + df["completed"].fillna(0) * 3.0
                + (df["avg_score"].fillna(0) / 100.0) * 1.0
            )

            # Build index maps
            learners = df["learner_email"].unique()
            courses = df["course_id"].unique()
            self._learner_index = {e: i for i, e in enumerate(learners)}
            self._course_index = list(courses)
            course_idx = {c: j for j, c in enumerate(courses)}

            # Build sparse learner × course matrix
            rows = df["learner_email"].map(self._learner_index).values
            cols = df["course_id"].map(course_idx).values
            mat = csr_matrix(
                (df["weight"].values, (rows, cols)),
                shape=(len(learners), len(courses)),
                dtype=np.float32,
            )

            # TruncatedSVD factorisation
            n_components = min(N_COMPONENTS, len(courses) - 1)
            self._svd = TruncatedSVD(n_components=n_components, random_state=42)
            self._learner_factors = self._svd.fit_transform(mat)   # (n_learners, k)
            self._course_factors = self._svd.components_.T          # (n_courses, k)

            # Store started courses per learner for filtering at inference time
            for email, grp in df.groupby("learner_email"):
                self._learner_courses[str(email)] = set(grp["course_id"])

            logger.info(
                "CourseRecommender trained. learners=%d  courses=%d  k=%d  "
                "explained_variance=%.3f",
                len(learners),
                len(courses),
                n_components,
                float(self._svd.explained_variance_ratio_.sum()),
            )

    # ── Inference ─────────────────────────────────────────────────────────────

    def recommend(self, learner_email: str, top_n: int = 5) -> list[dict]:
        """
        Return the top-N course recommendations for a learner.

        Courses the learner has already started are excluded. Unknown learners
        (not in training data) return an empty list rather than raising.

        Parameters
        ----------
        learner_email : str
            The learner's email address (must match silver_xapi_events).
        top_n : int
            Maximum number of recommendations to return.

        Returns
        -------
        list[dict]
            Each dict has keys: course_id (str), score (float).
            Sorted descending by score (most recommended first).
        """
        if self._svd is None:
            raise RuntimeError("Model not trained. Call .train(analytics) or use .from_analytics().")
        if learner_email not in self._learner_index:
            return []

        idx = self._learner_index[learner_email]
        learner_vec = self._learner_factors[idx]          # (k,)
        scores = self._course_factors @ learner_vec       # (n_courses,)

        started = self._learner_courses.get(learner_email, set())
        results = [
            {"course_id": self._course_index[j], "score": round(float(scores[j]), 4)}
            for j in np.argsort(-scores)
            if self._course_index[j] not in started
        ]
        return results[:top_n]

    # ── Diagnostics ───────────────────────────────────────────────────────────

    @property
    def is_trained(self) -> bool:
        return self._svd is not None

    @property
    def model_info(self) -> dict:
        """Return a summary dict for display in the UI."""
        return {
            "trained": self.is_trained,
            "n_learners": len(self._learner_index),
            "n_courses": len(self._course_index),
            "n_components": self._svd.n_components if self._svd else 0,
            "explained_variance_ratio": round(
                float(self._svd.explained_variance_ratio_.sum()), 4
            ) if self._svd else 0.0,
        }
