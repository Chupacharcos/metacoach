"""MetaCoach — Endpoints FastAPI"""

import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from metacoach import (
    analyze_wearables, analyze_blood,
    generate_weekly_plan, chat,
)

router = APIRouter(prefix="/metacoach")


class WearableData(BaseModel):
    sleep_hours: float = 7.5
    hrv: float = 65.0
    steps: int = 8000
    active_calories: int = 400
    resting_hr: int = 60


class BloodData(BaseModel):
    ferritina: Optional[float] = None
    hemoglobina: Optional[float] = None
    vitamina_d: Optional[float] = None
    glucosa: Optional[float] = None
    tsh: Optional[float] = None


class UserProfile(BaseModel):
    age: int = 30
    weight_kg: float = 75.0
    height_cm: float = 175.0
    sex: str = "h"  # h | m
    goal: str = "salud general"


class GeneratePlanRequest(BaseModel):
    session_id: str = ""
    profile: UserProfile
    wearable: WearableData
    blood: BloodData = BloodData()


class ChatRequest(BaseModel):
    session_id: str
    profile: UserProfile
    plan_context: str = ""
    message: str


@router.get("/health")
def health():
    return {"status": "ok", "service": "metacoach", "version": "1.0.0"}


@router.post("/generate-plan")
def generate_plan(req: GeneratePlanRequest):
    try:
        session_id = req.session_id or str(uuid.uuid4())

        wearable_analysis = analyze_wearables(req.wearable.model_dump())
        blood_analysis = analyze_blood(req.blood.model_dump(), req.profile.sex)

        result = generate_weekly_plan(
            req.profile.model_dump(),
            wearable_analysis,
            blood_analysis,
        )

        return {
            "session_id": session_id,
            "plan": result["plan"],
            "alertas": result["alertas"],
            "wearable_summary": wearable_analysis["resumen"],
            "blood_summary": blood_analysis["resumen"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat")
def coach_chat(req: ChatRequest):
    if not req.session_id:
        raise HTTPException(status_code=400, detail="session_id requerido")
    try:
        result = chat(
            req.session_id,
            req.profile.model_dump(),
            req.plan_context,
            req.message,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reference-ranges")
def reference_ranges():
    """Devuelve los rangos de referencia para los valores de analítica."""
    from metacoach import BLOOD_RANGES, HRV_ZONES, SLEEP_ZONES
    return {
        "blood": BLOOD_RANGES,
        "hrv_zones": HRV_ZONES,
        "sleep_zones": SLEEP_ZONES,
    }
