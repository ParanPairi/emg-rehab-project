from fastapi import APIRouter, HTTPException
from services.training_service import (
    start_training,
    stop_training,
    get_training_status,
)

router = APIRouter()

# Rest 제외 5가지 훈련 가능 동작
VALID_GESTURES = [
    "Fist",
    "Wrist Flexion",
    "Wrist Extension",
    "Elbow Flexion",
    "Elbow Extension",
]
VALID_COUNTS = [10, 20, 30, 50]


@router.post("/training/start/{user_id}")
def training_start(user_id: str, gesture: str, count: int):
    if gesture not in VALID_GESTURES:
        raise HTTPException(
            status_code=400,
            detail=f"훈련 가능한 동작이 아닙니다: {gesture} (가능: {VALID_GESTURES})",
        )
    if count not in VALID_COUNTS:
        raise HTTPException(
            status_code=400,
            detail=f"목표 횟수는 {VALID_COUNTS} 중 하나여야 합니다.",
        )

    status = start_training(user_id, gesture, count)
    return {"success": True, "message": "훈련을 시작합니다.", "status": status}


@router.post("/training/stop")
def training_stop():
    status = stop_training()
    return {"success": True, "message": "훈련을 종료합니다.", "status": status}


@router.get("/training/status")
def training_status():
    return get_training_status()