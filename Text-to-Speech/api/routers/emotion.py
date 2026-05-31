"""POST /emotion-analysis — text emotion detection."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from ..schemas.requests import EmotionAnalysisRequest
from ..schemas.responses import EmotionAnalysisResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/emotion-analysis", tags=["Emotion"])


def get_detector():
    from ..dependencies import get_emotion_detector
    return get_emotion_detector()


@router.post("", response_model=EmotionAnalysisResponse)
async def analyze_emotion(
    request: EmotionAnalysisRequest,
    detector=Depends(get_detector),
):
    """Detect emotion and intensity from text."""
    try:
        if request.per_sentence:
            sentence_results = detector.detect_sentences(request.text)
            sentences = [r.to_dict() for r in sentence_results]
            # Overall = most common emotion
            from collections import Counter
            most_common = Counter(r.emotion for r in sentence_results).most_common(1)
            overall = sentence_results[0] if sentence_results else None
        else:
            result = detector.detect_multilingual(request.text, request.language)
            sentences = None
            overall = result

        if overall is None:
            raise HTTPException(status_code=422, detail="Could not analyze empty text")

        return EmotionAnalysisResponse(
            emotion=overall.emotion,
            intensity=overall.intensity,
            confidence=overall.confidence,
            scores=overall.scores,
            sentences=sentences,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Emotion analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
