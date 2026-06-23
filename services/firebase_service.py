from firebase_admin import firestore

from core.firebase import firestore_db
from core.config import CNN_MODEL_PATH


def save_emg_log_to_firestore(
    user_id: str,
    gesture: str,
    confidence: float,
    feature_data: dict,
    raw_window_shape: tuple,
) -> str:
    doc_ref = (
        firestore_db.collection("users")
        .document(user_id)
        .collection("emgLogs")
        .document()
    )

    # ✅ numpy array를 list로 변환
    rms = feature_data["rms"]
    mav = feature_data["mav"]
    wl = feature_data["wl"]
    max_val = feature_data.get("max")
    var_val = feature_data.get("var")
    features = feature_data["features"]

    doc_ref.set({
        "userId": user_id,
        "gesture": gesture,
        "confidence": round(confidence, 4),

        "rmsAvg": float(feature_data["rms_avg"]),
        "mavAvg": float(feature_data["mav_avg"]),
        "wlAvg": float(feature_data["wl_avg"]),

        "rmsValues": [float(x) for x in rms],  # ✅ list이므로 직접 순회
        "mavValues": [float(x) for x in mav],  # ✅ list이므로 직접 순회
        "wlValues": [float(x) for x in wl],   # ✅ list이므로 직접 순회
        "features": [float(x) for x in features],  # ✅ list이므로 직접 순회
        
        # ✅ MAX, VAR 추가
        **({"maxValues": [float(x) for x in max_val]} if max_val is not None else {}),
        **({"varValues": [float(x) for x in var_val]} if var_val is not None else {}),

        "windowLength": int(raw_window_shape[0]),
        "channelCount": int(raw_window_shape[1]),

        "source": "fastapi-cnn",
        "modelType": "CNN",
        "modelFile": CNN_MODEL_PATH,

        "createdAt": firestore.SERVER_TIMESTAMP,
    })

    return doc_ref.id


def get_firestore_logs_by_user(user_id: str, limit: int = 20) -> list[dict]:
    docs = (
        firestore_db.collection("users")
        .document(user_id)
        .collection("emgLogs")
        .order_by("createdAt", direction=firestore.Query.DESCENDING)
        .limit(limit)
        .stream()
    )

    logs = []

    for doc in docs:
        item = doc.to_dict()
        item["id"] = doc.id

        created_at = item.get("createdAt")
        item["createdAtText"] = created_at.isoformat() if created_at else None

        logs.append(item)

    return logs

def save_personal_training_data(
    user_id: str,
    target_gesture: str,
    predicted_gesture: str,
    confidence: float,
    raw_window: list[list[float]],
    feature_data: dict,
) -> str:
    doc_ref = (
        firestore_db.collection("users")
        .document(user_id)
        .collection("personalTrainingData")
        .document()
    )

    window_length = len(raw_window)
    channel_count = len(raw_window[0]) if raw_window else 0
    
    # ✅ raw_window 타입을 완전히 Python native로 변환
    try:
        raw_window_clean = [[float(x) for x in row] for row in raw_window]
    except (TypeError, AttributeError):
        raw_window_clean = raw_window
    
    # ✅ feature_data에서 필요한 값만 추출 (numpy array 제거)
    rms_avg = float(feature_data.get("rms_avg", 0.0))
    mav_avg = float(feature_data.get("mav_avg", 0.0))
    wl_avg = float(feature_data.get("wl_avg", 0.0))
    
    # ✅ Firebase는 nested array를 허용하지 않으므로 flat 형식으로 저장
    # 2D 배열을 1D로 flatten: [[1,2,3], [4,5,6]] → [1,2,3,4,5,6]
    raw_window_flat = [float(x) for row in raw_window_clean for x in row]

    doc_ref.set({
        "userId": user_id,

        # 사용자가 선택한 훈련 동작 = 개인화 학습용 정답 라벨
        "targetGesture": target_gesture,

        # 현재 CNN이 예측한 결과
        "predictedGesture": predicted_gesture,
        "confidence": round(float(confidence), 4),

        # ✅ raw 데이터를 flat 형식으로 저장 (nested array 금지)
        "windowDataFlat": raw_window_flat,
        "windowLength": int(window_length),
        "channelCount": int(channel_count),

        # 분석용 특징값 (float만)
        "rmsAvg": rms_avg,
        "mavAvg": mav_avg,
        "wlAvg": wl_avg,

        "source": "app-training",
        "createdAt": firestore.SERVER_TIMESTAMP,
    })

    return doc_ref.id


def get_personal_training_count(user_id: str) -> int:
    docs = (
        firestore_db.collection("users")
        .document(user_id)
        .collection("personalTrainingData")
        .stream()
    )

    return sum(1 for _ in docs)


def update_personal_model_status(
    user_id: str,
    mode: str,
    has_personal_model: bool,
    training_data_count: int,
    required_data_count: int,
    is_training: bool,
    error: str | None = None,
):
    doc_ref = (
        firestore_db.collection("users")
        .document(user_id)
        .collection("modelStatus")
        .document("personalization")
    )

    data = {
        "mode": mode,
        "hasPersonalModel": has_personal_model,
        "trainingDataCount": training_data_count,
        "requiredDataCount": required_data_count,
        "isTraining": is_training,
        "lastUpdated": firestore.SERVER_TIMESTAMP,
    }

    if error:
        data["error"] = error

    doc_ref.set(data, merge=True)