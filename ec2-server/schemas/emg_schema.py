from pydantic import BaseModel, Field, validator
from typing import Optional


class EMGData(BaseModel):
    user_id: str = Field(..., min_length=1)
    
    # 🚨 수정: 단순히 리스트의 리스트라고만 하지 않고, 
    # 설명(description)과 함께 Pydantic 차원의 검증을 추가합니다.
    window_data: list[list[float]] = Field(
        ..., 
        min_length=1, 
        description="50 프레임 x 4 채널 크기의 2차원 EMG 데이터 배열"
    )

    # 개인화 학습용 정답 라벨
    # 예: Fist, Wrist Flexion, Wrist Extension
    target_gesture: Optional[str] = None

    # 🚨 [시니어 팁] Pydantic Validator를 통한 4채널 강력 검증
    @validator("window_data")
    def check_window_shape(cls, v):
        expected_frames = 25
        expected_channels = 4
        
        # 1. 프레임 수(행) 검사
        if len(v) != expected_frames:
            raise ValueError(f"데이터 프레임 수가 맞지 않습니다. 기대값: {expected_frames}, 실제값: {len(v)}")
        
        # 2. 채널 수(열) 검사 (첫 번째 프레임 기준)
        if len(v[0]) != expected_channels:
            raise ValueError(f"채널 수가 맞지 않습니다. 기대값: {expected_channels}, 실제값: {len(v[0])}")
            
        return v


class PrescriptionRequest(BaseModel):
    patient_name: str
    fatigue_slope: float
    comp_ratio: float
    days_to_target: int


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)