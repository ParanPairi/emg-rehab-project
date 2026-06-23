from fastapi import APIRouter, BackgroundTasks, HTTPException

from schemas.emg_schema import EMGData
from services.feature_service import validate_emg_window, extract_features
from services.cnn_model_service import predict_gesture_with_cnn
from services.firebase_service import (
    save_emg_log_to_firestore,
    save_personal_training_data,
)
from services.personalization_service import check_and_train_personal_model


router = APIRouter()


@router.post("/api/predict")
def predict_and_save(data: EMGData, background_tasks: BackgroundTasks):
    try:
        raw_window = validate_emg_window(data.window_data)

        feature_data = extract_features(raw_window)

        prediction = predict_gesture_with_cnn(
            raw_window,
            user_id=data.user_id,
        )

        gesture = prediction["gesture"]
        confidence = float(prediction["confidence"])

        firebase_doc_id = save_emg_log_to_firestore(
            user_id=data.user_id,
            gesture=gesture,
            confidence=confidence,
            feature_data=feature_data,
            raw_window_shape=raw_window.shape,
        )

        personal_training_doc_id = None

        if data.target_gesture is not None and data.target_gesture.strip():
            # ✅ raw_window를 완전히 Python native 타입으로 변환
            raw_window_native = [[float(x) for x in row] for row in raw_window.tolist()]
            
            personal_training_doc_id = save_personal_training_data(
                user_id=data.user_id,
                target_gesture=data.target_gesture.strip(),
                predicted_gesture=gesture,
                confidence=confidence,
                raw_window=raw_window_native,
                feature_data=feature_data,
            )

            background_tasks.add_task(
                check_and_train_personal_model,
                data.user_id,
            )

        return {
            "status": "success",
            "firebaseDocId": firebase_doc_id,
            "personalTrainingDocId": personal_training_doc_id,
            "user_id": data.user_id,
            "target_gesture": data.target_gesture,
            "gesture": gesture,
            "confidence": round(confidence, 4),
            "rmsAvg": round(feature_data["rms_avg"], 4),
            "mavAvg": round(feature_data["mav_avg"], 4),
            "wlAvg": round(feature_data["wl_avg"], 4),
            "modelType": "CNN",
            "modelInputShape": prediction.get("modelInputShape"),
            "db_status": "Saved to Firebase only",
        }

    except Exception as e:
        import traceback
        print("CNN prediction save error:", e)
        print("Full traceback:")
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail={
                "message": "CNN 예측 저장 실패",
                "error": str(e),
            },
        )