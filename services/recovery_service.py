import numpy as np
from sklearn.linear_model import LinearRegression
from firebase_admin import firestore

from core.firebase import firestore_db
from core.config import TARGET_RMS


def predict_recovery_by_user(user_id: str) -> dict:
    docs = (
        firestore_db.collection("users")
        .document(user_id)
        .collection("emgLogs")
        .order_by("createdAt", direction=firestore.Query.DESCENDING)
        .limit(14)
        .stream()
    )

    rms_history = []

    for doc in docs:
        item = doc.to_dict()
        rms_avg = item.get("rmsAvg")

        if rms_avg is not None:
            try:
                rms_history.append(float(rms_avg))
            except Exception:
                pass

    rms_history = list(reversed(rms_history))

    if len(rms_history) < 3:
        return {
            "user_id": user_id,
            "status": "not_enough_data",
            "message": "회복 추세를 예측하려면 최소 3개 이상의 EMG 기록이 필요합니다.",
            "data_count": len(rms_history),
            "current_rms": None,
            "target_rms": TARGET_RMS,
            "daily_recovery_slope": None,
            "remaining_days": None,
            "estimated_target_day": None,
            "rms_history": [round(x, 4) for x in rms_history],
        }

    past_days = len(rms_history)
    days_x = np.arange(1, past_days + 1).reshape(-1, 1)

    lr_model = LinearRegression()
    lr_model.fit(days_x, rms_history)

    current_rms = float(rms_history[-1])
    predicted_slope = float(lr_model.coef_[0])

    if predicted_slope > 0:
        intercept = float(lr_model.intercept_)
        days_to_target_total = (TARGET_RMS - intercept) / predicted_slope
        remaining_days = int(days_to_target_total - past_days)

        if remaining_days < 0:
            remaining_days = 0

        status = "recovering"
        message = f"현재 추세를 유지하면 약 {remaining_days}일 후 목표 근활성도에 도달할 것으로 예상됩니다."

    else:
        days_to_target_total = None
        remaining_days = None
        status = "stagnant"
        message = "회복 추세가 정체되거나 하락하고 있어 운동 강도 조절이 필요합니다."

    return {
        "user_id": user_id,
        "status": status,
        "message": message,
        "data_count": past_days,
        "current_rms": round(current_rms, 4),
        "target_rms": TARGET_RMS,
        "daily_recovery_slope": round(predicted_slope, 4),
        "remaining_days": remaining_days,
        "estimated_target_day": int(days_to_target_total) if days_to_target_total else None,
        "rms_history": [round(x, 4) for x in rms_history],
    }