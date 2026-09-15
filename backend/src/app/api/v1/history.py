from fastapi import APIRouter, HTTPException
from typing import List
from src.app.db.repository import get_recent_analyses, get_analysis_by_id
from pydantic import BaseModel, Field
from typing import Optional, Any, Dict

router = APIRouter()


class HistorySummary(BaseModel):
    """Lightweight summary of a past analysis (for the history list)."""
    id: str
    image_name: str
    prediction: str
    is_ai_generated: bool
    calibrated_confidence: float
    confidence_percentage: str
    created_at: Optional[str] = None


class HistoryListResponse(BaseModel):
    total: int
    results: List[HistorySummary]


@router.get(
    "/history",
    response_model=HistoryListResponse,
    summary="Get Recent Analysis History",
    description="Returns the most recent image analysis results stored in the database."
)
async def get_history(limit: int = 20):
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100.")
    records = get_recent_analyses(limit=limit)
    return {"total": len(records), "results": records}


@router.get(
    "/history/{analysis_id}",
    summary="Get Analysis by ID",
    description="Returns the full analysis result for a given database record ID."
)
async def get_history_item(analysis_id: str):
    record = get_analysis_by_id(analysis_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Analysis with id '{analysis_id}' not found.")
    return record
