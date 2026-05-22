from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import duckdb
import joblib
import numpy as np
import os
import pandas as pd

# ============================================================
# APP CONFIG
# ============================================================

app = FastAPI(title="Recommendation Engine API", version="1.0")

MODEL_PATH = "phase2_lgbm_v3.pkl"
TABLE_DIR = "recommender_api_tables"

# ============================================================
# LOAD MODEL + DUCKDB TABLES
# ============================================================

model = joblib.load(MODEL_PATH)

con = duckdb.connect()
con.execute("SET memory_limit='2GB'")
con.execute("SET threads=1")

API_TABLES = [
    "post_meta",
    "phase1_view_top_neighbors",
    "active_post_fallback_base",
    "recent_seen_posts",
    "post_view_stats",
    "post_recent_view_stats",
    "post_dwell_agg",
    "post_interaction_agg",
    "ads_new_features",
    "accounts_features",
    "user_cat_affinity",
    "user_city_affinity",
]

for table in API_TABLES:
    path = os.path.join(TABLE_DIR, f"{table}.parquet")
    con.execute(f"""
        CREATE OR REPLACE VIEW {table} AS
        SELECT * FROM read_parquet('{path}')
    """)

FEATURE_COLS = [
    "co_view_score", "transition_count", "graph_unique_users", "in_graph",
    "same_category", "same_city", "total_views", "unique_viewers",
    "avg_view_quality", "high_quality_view_rate", "bounce_rate",
    "recent_views_3d", "recent_unique_3d", "avg_dwell_seconds",
    "avg_scroll_depth", "target_msg_count", "target_like_count",
    "target_fav_count", "interaction_strength", "ad_views",
    "times_renewed", "is_active", "post_age_days",
    "user_cat_affinity", "user_city_affinity", "user_messages_sent",
    "user_messages_received", "user_page_views", "user_posts",
    "user_membership_level",
]


# ============================================================
# REQUEST MODEL
# ============================================================

class RecommendRequest(BaseModel):
    user_id: int
    source_post_id: int
    top_k: int = 10


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():
    return {"message": "Recommendation API working successfully"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "features_count": len(FEATURE_COLS),
        "tables_loaded": len(API_TABLES),
    }


# ============================================================
# RECOMMENDATION LOGIC
# ============================================================

def generate_candidates_phase1_sql(
    user_id: int,
    source_post_id: int,
    top_k_candidates: int = 300
) -> pd.DataFrame:

    query = f"""
    WITH source_meta AS (
        SELECT city_id AS src_city_id, cat_id AS src_cat_id
        FROM post_meta
        WHERE post_id = {source_post_id}
    ),

    graph_cands AS (
        SELECT
            target_post_id AS post_id,
            CASE
                WHEN MAX(co_view_score) OVER () > 0
                THEN co_view_score / MAX(co_view_score) OVER ()
                ELSE 0.0
            END AS phase1_score,
            1 AS priority_group
        FROM phase1_view_top_neighbors
        WHERE source_post_id = {source_post_id}
    ),

    fallback_cands AS (
        SELECT
            f.post_id,
            (
                0.45 * CASE WHEN f.cat_id = sm.src_cat_id THEN 1.0 ELSE 0.0 END +
                0.20 * CASE WHEN f.city_id = sm.src_city_id THEN 1.0 ELSE 0.0 END +
                0.20 * f.recency_score +
                0.15 * f.popularity_score
            ) AS phase1_score,
            2 AS priority_group
        FROM active_post_fallback_base f
        CROSS JOIN source_meta sm
        WHERE f.post_id <> {source_post_id}
        LIMIT 200
    ),

    merged AS (
        SELECT
            post_id,
            SUM(phase1_score) AS phase1_score,
            MIN(priority_group) AS priority_group
        FROM (
            SELECT * FROM graph_cands
            UNION ALL
            SELECT * FROM fallback_cands
        )
        GROUP BY post_id
    )

    SELECT
        m.post_id,
        m.phase1_score
    FROM merged m
    LEFT JOIN recent_seen_posts rsp
        ON rsp.user_id = {user_id}
       AND rsp.post_id = m.post_id
    WHERE rsp.post_id IS NULL
      AND m.post_id <> {source_post_id}
    ORDER BY priority_group ASC, phase1_score DESC, post_id
    LIMIT {top_k_candidates}
    """

    return con.execute(query).df()


def score_candidates(
    user_id: int,
    source_post_id: int,
    candidates: pd.DataFrame,
    top_k: int
) -> pd.DataFrame:

    if candidates.empty:
        return pd.DataFrame(columns=[
            "post_id", "final_score", "phase2_score", "phase1_score", "in_graph"
        ])

    post_ids_str = ", ".join(str(int(p)) for p in candidates["post_id"].tolist())

    feat = con.execute(f"""
        WITH tgt AS (
            SELECT post_id, city_id AS tgt_city, cat_id AS tgt_cat, post_date
            FROM post_meta
            WHERE post_id IN ({post_ids_str})
        ),

        src AS (
            SELECT city_id AS src_city, cat_id AS src_cat
            FROM post_meta
            WHERE post_id = {source_post_id}
        ),

        max_dt AS (
            SELECT MAX(CAST(post_date AS DATE)) AS md
            FROM post_meta
        )

        SELECT
            tgt.post_id,

            COALESCE(n.co_view_score, 0.0) AS co_view_score,
            COALESCE(n.transition_count, 0) AS transition_count,
            COALESCE(n.unique_users, 0) AS graph_unique_users,
            CASE WHEN n.co_view_score IS NOT NULL THEN 1 ELSE 0 END AS in_graph,

            CASE WHEN tgt.tgt_cat = src.src_cat THEN 1 ELSE 0 END AS same_category,
            CASE WHEN tgt.tgt_city = src.src_city THEN 1 ELSE 0 END AS same_city,

            COALESCE(pvs.total_views, 0) AS total_views,
            COALESCE(pvs.unique_viewers, 0) AS unique_viewers,
            COALESCE(pvs.avg_view_quality, 0.0) AS avg_view_quality,
            COALESCE(pvs.high_quality_view_rate, 0.0) AS high_quality_view_rate,
            COALESCE(pvs.bounce_rate, 0.0) AS bounce_rate,

            COALESCE(prvs.recent_views_3d, 0) AS recent_views_3d,
            COALESCE(prvs.recent_unique_viewers_3d, 0) AS recent_unique_3d,

            COALESCE(pda.avg_dwell_seconds, 0.0) AS avg_dwell_seconds,
            COALESCE(pda.avg_scroll_depth, 0.0) AS avg_scroll_depth,

            COALESCE(pia.msg_cnt, 0) AS target_msg_count,
            COALESCE(pia.like_cnt, 0) AS target_like_count,
            COALESCE(pia.fav_cnt, 0) AS target_fav_count,
            COALESCE(pia.interaction_strength, 0.0) AS interaction_strength,

            COALESCE(anf.ad_views, 0) AS ad_views,
            COALESCE(anf.times_renewed, 0) AS times_renewed,
            COALESCE(anf.is_active, 0) AS is_active,

            CASE
                WHEN tgt.post_date IS NULL THEN 9999
                ELSE date_diff('day', CAST(tgt.post_date AS DATE), (SELECT md FROM max_dt))
            END AS post_age_days,

            COALESCE(uca.cat_affinity, 0.0) AS user_cat_affinity,
            COALESCE(ucit.city_affinity, 0.0) AS user_city_affinity,

            COALESCE(af.messages_sent, 0) AS user_messages_sent,
            COALESCE(af.messages_received, 0) AS user_messages_received,
            COALESCE(af.page_views, 0) AS user_page_views,
            COALESCE(af.number_of_posts, 0) AS user_posts,
            COALESCE(af.membership_level, 0) AS user_membership_level

        FROM tgt
        CROSS JOIN src
        LEFT JOIN phase1_view_top_neighbors n
            ON n.source_post_id = {source_post_id}
           AND n.target_post_id = tgt.post_id
        LEFT JOIN post_view_stats pvs ON tgt.post_id = pvs.post_id
        LEFT JOIN post_recent_view_stats prvs ON tgt.post_id = prvs.post_id
        LEFT JOIN post_dwell_agg pda ON tgt.post_id = pda.post_id
        LEFT JOIN post_interaction_agg pia ON tgt.post_id = pia.post_id
        LEFT JOIN ads_new_features anf ON tgt.post_id = anf.post_id
        LEFT JOIN accounts_features af ON af.user_id = {user_id}
        LEFT JOIN user_cat_affinity uca
            ON uca.user_id = {user_id}
           AND tgt.tgt_cat = uca.cat_id
        LEFT JOIN user_city_affinity ucit
            ON ucit.user_id = {user_id}
           AND tgt.tgt_city = ucit.city_id
        WHERE COALESCE(anf.is_spammer, 0) = 0
    """).df()

    if feat.empty:
        return pd.DataFrame(columns=[
            "post_id", "final_score", "phase2_score", "phase1_score", "in_graph"
        ])

    feat["phase2_score"] = model.predict_proba(
        feat[FEATURE_COLS].astype(np.float32)
    )[:, 1]

    feat = feat.merge(
        candidates[["post_id", "phase1_score"]],
        on="post_id",
        how="left"
    )

    feat["phase1_score"] = feat["phase1_score"].fillna(0)

    feat["final_score"] = (
        0.70 * feat["phase2_score"] +
        0.30 * feat["phase1_score"]
    )

    return (
        feat[["post_id", "final_score", "phase2_score", "phase1_score", "in_graph"]]
        .sort_values("final_score", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )


def rerank_phase2(
    user_id: int,
    source_post_id: int,
    top_k: int = 10
) -> pd.DataFrame:

    candidates = generate_candidates_phase1_sql(
        user_id=user_id,
        source_post_id=source_post_id,
        top_k_candidates=300
    )

    return score_candidates(
        user_id=user_id,
        source_post_id=source_post_id,
        candidates=candidates,
        top_k=top_k
    )


@app.post("/recommend")
def recommend(req: RecommendRequest):
    try:
        recs = rerank_phase2(
            user_id=req.user_id,
            source_post_id=req.source_post_id,
            top_k=req.top_k
        )

        return {
            "user_id": req.user_id,
            "source_post_id": req.source_post_id,
            "top_k": req.top_k,
            "count": len(recs),
            "recommendations": recs.to_dict(orient="records"),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))