import time
import numpy as np

from services.cnn_model_service import predict_gesture_with_cnn
from services.feature_service import extract_features
from services.firebase_service import save_emg_log_to_firestore
from core.config import WINDOW_SIZE

window_buffer = []
latest_prediction = None


def add_emg(values, user_id="default"):
    global window_buffer, latest_prediction

    if len(values) != 5:
        return None

    values = [float(v) for v in values]

    window_buffer.append(values)

    if len(window_buffer) > WINDOW_SIZE:
        window_buffer = window_buffer[-WINDOW_SIZE:]

    if len(window_buffer) < WINDOW_SIZE:
        return None

    raw_window = np.array(window_buffer, dtype=np.float32)

    prediction = predict_gesture_with_cnn(
        raw_window=raw_window,
        user_id=user_id,
    )

    feature_data = extract_features(raw_window)

    doc_id = save_emg_log_to_firestore(
        user_id=user_id,
        gesture=prediction["gesture"],
        confidence=prediction["confidence"],
        feature_data=feature_data,
        raw_window_shape=raw_window.shape,
    )

    latest_prediction = {
        "predicted_at": time.time(),
        "doc_id": doc_id,
        "gesture": prediction["gesture"],
        "confidence": prediction["confidence"],
    }

    return latest_prediction


def get_stream_status():
    return {
        "buffer_size": len(window_buffer),
        "latest_prediction": latest_prediction,
    }