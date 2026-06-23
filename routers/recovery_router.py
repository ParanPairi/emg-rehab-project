from fastapi import APIRouter

from services.recovery_service import predict_recovery_by_user


router = APIRouter()


@router.get("/api/recovery/{user_id}")
def predict_recovery(user_id: str):
    return predict_recovery_by_user(user_id)