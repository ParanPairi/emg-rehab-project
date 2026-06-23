from pydantic import BaseModel, Field
from typing import Optional


class EMGData(BaseModel):
    user_id: str = Field(..., min_length=1)
    window_data: list[list[float]] = Field(..., min_length=1)

    # 개인화 학습용 정답 라벨
    # 예: Fist, Wrist Flexion, Wrist Extension
    target_gesture: Optional[str] = None


class PrescriptionRequest(BaseModel):
    patient_name: str
    fatigue_slope: float
    comp_ratio: float
    days_to_target: int


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)