from fastapi import APIRouter

from services.firebase_service import get_firestore_logs_by_user


router = APIRouter()


@router.get("/api/firebase/logs/{user_id}")
def get_firebase_logs(user_id: str):
    return get_firestore_logs_by_user(user_id=user_id, limit=20)