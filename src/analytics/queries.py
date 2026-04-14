"""
Sprint 2 — Task 2.1: FabricAnalytics query layer.

All methods query Gold/Silver/Dim tables on the Fabric Lakehouse SQL analytics
endpoint via pyodbc. The SQL dialect is T-SQL (Fabric's SQL engine is compatible
with SQL Server syntax).

Usage
-----
from src.analytics.queries import FabricAnalytics
analytics = FabricAnalytics.from_env()
kpis = analytics.get_overview_kpis()
"""
from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd

from src.fabric.sql_client import FabricSQLClient

logger = logging.getLogger(__name__)


class FabricAnalytics:
    """
    High-level analytics queries against Fabric Lakehouse gold/silver tables.

    All public methods return either a list[dict] (tabular) or a dict (single row).
    The Streamlit dashboard accesses this class exclusively — no raw SQL leaks out.
    """

    def __init__(self, client: FabricSQLClient) -> None:
        self._db = client

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_env(cls) -> "FabricAnalytics":
        """Create a FabricAnalytics instance from environment variables."""
        client = FabricSQLClient.from_env()
        return cls(client)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _rows(self, sql: str, params: list | None = None) -> list[dict[str, Any]]:
        df = self._db.query(sql, params)
        return df.to_dict(orient="records")

    def _one(self, sql: str, params: list | None = None) -> dict[str, Any]:
        row = self._db.query_one(sql, params)
        return row or {}

    def _df(self, sql: str, params: list | None = None) -> pd.DataFrame:
        return self._db.query(sql, params)

    # ── 1. Overview KPIs ──────────────────────────────────────────────────────

    def get_overview_kpis(self) -> dict[str, Any]:
        """
        Returns a single dict with four headline metrics:
          total_learners, active_this_month, avg_completion_rate, total_courses
        """
        sql = """
            SELECT
                (SELECT COUNT(DISTINCT learner_email)
                 FROM silver_xapi_events)                                       AS total_learners,

                (SELECT COUNT(DISTINCT learner_email)
                 FROM silver_xapi_events
                 WHERE ts >= DATEADD(month, -1, GETDATE()))                     AS active_this_month,

                (SELECT ROUND(AVG(CAST(completion_rate_pct AS FLOAT)), 1)
                 FROM gold_course_completion
                 WHERE enrolled >= 5)                                           AS avg_completion_rate,

                (SELECT COUNT(*) FROM dim_courses)                              AS total_courses
        """
        return self._one(sql)

    # ── 2. Completion funnel ──────────────────────────────────────────────────

    def get_completion_funnel(self, course_id: str) -> list[dict[str, Any]]:
        """
        Module-by-module completion data for a single course.

        Returns list of dicts with keys:
          module_id, learners_reached, learners_completed,
          module_completion_pct, avg_quiz_score, low_score_count
        """
        sql = """
            SELECT
                module_id,
                learners_reached,
                learners_completed,
                module_completion_pct,
                avg_quiz_score,
                low_score_count
            FROM gold_module_dropoff
            WHERE course_id = ?
            ORDER BY module_id
        """
        return self._rows(sql, [course_id])

    # ── 3. Engagement heatmap ────────────────────────────────────────────────

    def get_engagement_heatmap(self, course_id: str) -> list[dict[str, Any]]:
        """
        Event counts by module × hour-of-day for a course.

        Returns list of dicts with keys:
          module_id, hour_of_day, event_count, unique_learners
        """
        sql = """
            SELECT
                module_id,
                hour_of_day,
                SUM(event_count)       AS event_count,
                SUM(unique_learners)   AS unique_learners
            FROM gold_engagement_hourly
            WHERE course_id = ?
            GROUP BY module_id, hour_of_day
            ORDER BY module_id, hour_of_day
        """
        return self._rows(sql, [course_id])

    # ── 4. Department skill gaps ──────────────────────────────────────────────

    def get_department_skill_gaps(self, department: str) -> list[dict[str, Any]]:
        """
        Skills ranked by gap (target 3.5 minus actual proficiency) for a department.

        Returns list of dicts with keys:
          skill_name, avg_proficiency, target_proficiency, skill_gap, learner_count
        """
        sql = """
            SELECT
                skill_name,
                avg_proficiency,
                target_proficiency,
                skill_gap,
                learner_count
            FROM gold_department_skills
            WHERE department = ?
            ORDER BY skill_gap DESC
        """
        return self._rows(sql, [department])

    # ── 5. Course ranking ────────────────────────────────────────────────────

    def get_course_ranking(self) -> list[dict[str, Any]]:
        """
        All courses ranked by completion rate, enriched with department and format.

        Returns list of dicts with keys:
          course_id, title, department, course_format, skill_level,
          enrolled, completed, completion_rate_pct, avg_score
        """
        sql = """
            SELECT
                course_id,
                title,
                department,
                course_format,
                skill_level,
                enrolled,
                completed,
                completion_rate_pct,
                avg_score
            FROM gold_course_completion
            WHERE enrolled >= 3
            ORDER BY completion_rate_pct DESC
        """
        return self._rows(sql)

    # ── 6. At-risk learners ──────────────────────────────────────────────────

    def get_at_risk_learners(self, course_id: str) -> list[dict[str, Any]]:
        """
        Learners inactive for >7 days who haven't completed the course.

        Returns list of dicts with keys:
          learner_email, learner_name, department, modules_started,
          last_activity, days_inactive, avg_score
        """
        sql = """
            SELECT
                s.learner_email,
                s.learner_name,
                s.department,
                COUNT(DISTINCT s.module_id)          AS modules_started,
                MAX(s.ts)                            AS last_activity,
                DATEDIFF(day, MAX(s.ts), GETDATE())  AS days_inactive,
                ROUND(AVG(s.score), 3)               AS avg_score
            FROM silver_xapi_events s
            WHERE s.course_id = ?
              AND s.learner_email NOT IN (
                  SELECT learner_email
                  FROM silver_xapi_events
                  WHERE course_id = ?
                    AND verb = 'completed'
                    AND course_completed = 1
              )
            GROUP BY s.learner_email, s.learner_name, s.department
            HAVING DATEDIFF(day, MAX(s.ts), GETDATE()) > 7
            ORDER BY days_inactive DESC
        """
        return self._rows(sql, [course_id, course_id])

    # ── 7. Completion by format ──────────────────────────────────────────────

    def get_completion_by_format(self) -> list[dict[str, Any]]:
        """
        Average completion rate grouped by course format.

        Returns list of dicts with keys:
          course_format, num_courses, avg_completion_rate, avg_score
        """
        sql = """
            SELECT
                course_format,
                COUNT(*)                                 AS num_courses,
                ROUND(AVG(CAST(completion_rate_pct AS FLOAT)), 1) AS avg_completion_rate,
                ROUND(AVG(CAST(avg_score AS FLOAT)), 3)  AS avg_score
            FROM gold_course_completion
            WHERE enrolled >= 5
            GROUP BY course_format
            ORDER BY avg_completion_rate DESC
        """
        return self._rows(sql)

    # ── 8. Distinct departments ──────────────────────────────────────────────

    def get_departments(self) -> list[str]:
        """Return sorted list of distinct department names (for UI dropdowns)."""
        sql = "SELECT DISTINCT department FROM dim_learners ORDER BY department"
        df = self._df(sql)
        return df["department"].tolist() if not df.empty else []

    # ── 9. Course list ───────────────────────────────────────────────────────

    def get_courses(self, department: str | None = None) -> list[dict[str, Any]]:
        """
        Return course_id + title pairs, optionally filtered by department.

        Returns list of dicts with keys: course_id, title, department
        """
        if department:
            sql = """
                SELECT course_id, title, department
                FROM dim_courses
                WHERE department = ?
                ORDER BY title
            """
            return self._rows(sql, [department])
        else:
            sql = """
                SELECT course_id, title, department
                FROM dim_courses
                ORDER BY department, title
            """
            return self._rows(sql)

    # ── 10. Course stats (used by search indexer) ────────────────────────────

    def get_course_stats(self, course_id: str) -> dict[str, Any] | None:
        """Fetch completion_rate_pct and avg_score for a single course."""
        sql = """
            SELECT completion_rate_pct, avg_score
            FROM gold_course_completion
            WHERE course_id = ?
        """
        return self._db.query_one(sql, [course_id])

    def get_course_detail(self, course_id: str) -> dict[str, Any] | None:
        """Full course row from gold_course_completion + dim_courses."""
        sql = """
            SELECT g.*, d.description, d.skills_taught, d.num_modules, d.module_titles
            FROM gold_course_completion g
            JOIN dim_courses d ON g.course_id = d.course_id
            WHERE g.course_id = ?
        """
        return self._db.query_one(sql, [course_id])

    def get_courses_by_department(self, department: str) -> list[dict[str, Any]]:
        """Course list with analytics data, filtered by department."""
        sql = """
            SELECT g.course_id, g.title, g.skill_level, g.course_format,
                   g.enrolled, g.completion_rate_pct, g.avg_score
            FROM gold_course_completion g
            WHERE g.department = ?
            ORDER BY g.completion_rate_pct DESC
        """
        return self._rows(sql, [department])

    # ── Training data for ML model ────────────────────────────────────────────

    def get_interaction_matrix_data(self) -> pd.DataFrame:
        """
        One row per (learner_email, course_id) with interaction signals.

        Used to build the collaborative-filtering interaction matrix.
        Columns: learner_email, course_id, event_count, completed, avg_score
        """
        sql = """
            SELECT
                learner_email,
                course_id,
                COUNT(*)                                                        AS event_count,
                MAX(CASE WHEN course_completed = 1 THEN 1 ELSE 0 END)          AS completed,
                ROUND(AVG(CASE WHEN verb = 'scored' THEN score ELSE NULL END), 4) AS avg_score
            FROM silver_xapi_events
            GROUP BY learner_email, course_id
            HAVING COUNT(*) >= 2
        """
        return self._df(sql)

    def get_training_features(self) -> pd.DataFrame:
        """
        Build the feature matrix for dropout prediction from silver_xapi_events.

        Columns: learner_email, course_id, total_events, modules_touched,
                 avg_score, active_days, span_days, completed (label)
        """
        sql = """
            SELECT
                learner_email,
                course_id,
                COUNT(*)                              AS total_events,
                COUNT(DISTINCT module_id)             AS modules_touched,
                ROUND(AVG(CASE WHEN verb = 'scored' THEN score ELSE NULL END), 4)
                                                      AS avg_score,
                COUNT(DISTINCT CAST(ts AS DATE))      AS active_days,
                DATEDIFF(day, MIN(ts), MAX(ts))       AS span_days,
                MAX(CASE WHEN verb = 'completed'
                          AND course_completed = 1
                     THEN 1 ELSE 0 END)               AS completed
            FROM silver_xapi_events
            GROUP BY learner_email, course_id
            HAVING COUNT(*) >= 5
        """
        return self._df(sql)
