import logging
from typing import Optional
from src.app.db.supabase_client import get_supabase

logger = logging.getLogger(__name__)

TABLE = "analysis_results"


def save_analysis(result: dict) -> Optional[str]:
    """
    Saves a single analysis result to Supabase.
    Returns the generated UUID (id) on success, or None on failure.
    """
    supabase = get_supabase()
    if supabase is None:
        logger.warning("Supabase not available — skipping DB save.")
        return None

    # Build the row — store complex objects as JSON (Supabase handles dicts as JSONB)
    row = {
        "image_name": result.get("image_name", "unknown"),
        "prediction": result.get("prediction", ""),
        "is_ai_generated": result.get("is_ai_generated", False),
        "raw_probability_ai": float(result.get("raw_probability_ai", 0.0)),
        "calibrated_confidence": float(result.get("calibrated_confidence", 0.0)),
        "confidence_percentage": str(result.get("confidence_percentage", "0%")),
        "convnext_features_summary": result.get("convnext_features_summary"),
        "explanation": result.get("explanation"),
        "gradcam_heatmap": result.get("gradcam_heatmap"),
        "generator_attribution": result.get("generator_attribution"),
        "metadata_provenance": result.get("metadata_provenance"),
    }

    try:
        response = supabase.table(TABLE).insert(row).execute()
        if response.data:
            record_id = response.data[0].get("id")
            logger.info(f"Analysis saved to Supabase with id: {record_id}")
            return record_id
        return None
    except Exception as e:
        logger.error(f"Failed to save analysis to Supabase: {e}")
        return None


def get_recent_analyses(limit: int = 20) -> list:
    """
    Fetches the most recent analysis results from Supabase.
    Returns a list of records (dicts), or empty list on failure.
    """
    supabase = get_supabase()
    if supabase is None:
        return []

    try:
        response = (
            supabase.table(TABLE)
            .select("id, image_name, prediction, is_ai_generated, calibrated_confidence, confidence_percentage, created_at")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return response.data or []
    except Exception as e:
        logger.error(f"Failed to fetch recent analyses: {e}")
        return []


def get_analysis_by_id(analysis_id: str) -> Optional[dict]:
    """
    Fetches a single full analysis result by its UUID.
    Returns the record dict or None if not found.
    """
    supabase = get_supabase()
    if supabase is None:
        return None

    try:
        response = (
            supabase.table(TABLE)
            .select("*")
            .eq("id", analysis_id)
            .single()
            .execute()
        )
        return response.data
    except Exception as e:
        logger.error(f"Failed to fetch analysis {analysis_id}: {e}")
        return None
