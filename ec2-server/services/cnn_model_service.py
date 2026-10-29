import os
import numpy as np
import joblib
from tensorflow.keras.models import load_model

from core.config import (
    CNN_MODEL_PATH,  # ⚠️ 주의: 이 경로에 있는 모델도 4채널(50x4)용 모델이어야 합니다.
    GESTURE_LABELS,
)

# 1️⃣ 4채널용 스케일러 파일 경로로 수정
SCALER_RAW_PATH = "/home/ubuntu/capstone_ai/model/scaler_raw_4ch_env.pkl"
SCALER_FEAT_PATH = "/home/ubuntu/capstone_ai/model/scaler_feat_4ch_env.pkl"

loaded_user_models = {}

_scaler_raw = None
_scaler_feat = None


def _get_scalers():
    global _scaler_raw, _scaler_feat
    if _scaler_raw is None:
        _scaler_raw = joblib.load(SCALER_RAW_PATH)
        _scaler_feat = joblib.load(SCALER_FEAT_PATH)
    return _scaler_raw, _scaler_feat


def get_user_model(user_id: str):
    """
    유저 ID에 맞는 모델을 메모리에서 가져오거나, 없으면 디스크에서 로드합니다.
    """
    if user_id in loaded_user_models:
        return loaded_user_models[user_id]

    user_model_path = f"models/users/user_{user_id}.h5"

    if os.path.exists(user_model_path):
        print(f"📦 [{user_id}] 디스크에서 개인화 모델을 로드합니다...")
        model = load_model(user_model_path, compile=False)
    else:
        print(f"📦 [{user_id}] 개인화 모델이 없어 기본(Base) 모델을 로드합니다...")
        model = load_model(CNN_MODEL_PATH, compile=False)

    loaded_user_models[user_id] = model
    return model


def _preprocess_window(raw_window: np.ndarray):
    """
    캘리브레이션과 동일한 전처리 파이프라인:
    1. 베이스라인 영점 조절 (최솟값을 0으로 clip)
    2. 특징 추출 (RMS, MAV, max, var) → (4종류 × 4채널) = 16차원 벡터
    3. 스케일러 적용
    반환: (x_raw_scaled, x_feat_scaled)
    """
    # 🚨 [시니어 팁] 입력 데이터가 정확히 4채널인지 방어하는 로직 추가
    n_channels = raw_window.shape[1]
    if n_channels != 4:
        raise ValueError(f"입력 데이터 채널 에러: 4채널 데이터를 기대했으나, {n_channels}채널이 입력되었습니다. 형태: {raw_window.shape}")

    # 1) 베이스라인 영점 조절
    baseline = np.min(raw_window, axis=0)
    adj_window = np.clip(raw_window - baseline, a_min=0, a_max=None)

    # 2) 특징 추출 (4채널에 대해 연산하므로 각각 사이즈가 4가 됨)
    rms = np.sqrt(np.mean(adj_window ** 2, axis=0))
    mav = np.mean(np.abs(adj_window), axis=0)
    max_val = np.max(adj_window, axis=0)
    var_val = np.var(adj_window, axis=0)
    
    # 2️⃣ 기존 (20,) 벡터에서 4채널에 맞춰 (16,) 차원 벡터로 결합됨
    features = np.concatenate([rms, mav, max_val, var_val])  # (16,)

    # 3) 스케일러 적용
    scaler_raw, scaler_feat = _get_scalers()
    
    # 원시 데이터(raw) 스케일링
    x_raw_scaled = scaler_raw.transform(adj_window.reshape(-1, n_channels)).reshape(adj_window.shape)
    # 특징 데이터(feat) 스케일링
    x_feat_scaled = scaler_feat.transform(features.reshape(1, -1))[0]

    return x_raw_scaled, x_feat_scaled


def _to_probability_array(model_output) -> np.ndarray:
    probs = np.array(model_output, dtype=float)
    if probs.ndim > 1:
        probs = probs[0]
    if np.any(probs < 0) or np.any(probs > 1) or not np.isclose(np.sum(probs), 1.0, atol=0.05):
        exp_values = np.exp(probs - np.max(probs))
        probs = exp_values / np.sum(exp_values)
    return probs

def predict_gesture_with_cnn(raw_window: np.ndarray, user_id: str = "default") -> dict:
    input_data = raw_window.astype("float32")

    raw_min = float(np.min(input_data))
    raw_max = float(np.max(input_data))
    raw_mean = float(np.mean(input_data))
    raw_std = float(np.std(input_data))

    x_raw_scaled, x_feat_scaled = _preprocess_window(input_data)

    x_raw_input = np.expand_dims(x_raw_scaled, axis=0)
    x_feat_input = np.expand_dims(x_feat_scaled, axis=0)

    current_model = get_user_model(user_id)
    model_input_shape = current_model.input_shape

    # ⭐ predict() 대신 직접 호출 (실시간 단건 추론에 훨씬 빠름)
    model_output = current_model([x_raw_input, x_feat_input], training=False).numpy()

    raw_model_output = np.array(model_output, dtype=float)
    if raw_model_output.ndim > 1:
        raw_model_output_flat = raw_model_output[0].tolist()
    else:
        raw_model_output_flat = raw_model_output.tolist()

    probs = _to_probability_array(model_output)
    best_idx = int(np.argmax(probs))

    if best_idx >= len(GESTURE_LABELS):
        raise ValueError(
            f"모델 출력 클래스 수와 gesture_labels 개수가 맞지 않습니다. "
            f"best_idx={best_idx}, labels={len(GESTURE_LABELS)}, probs_len={len(probs)}"
        )

    gesture = GESTURE_LABELS[best_idx]
    confidence = float(probs[best_idx] * 100)

    return {
        "gesture": gesture,
        "confidence": confidence,
        "probabilities": [float(x) for x in probs.tolist()],
        "rawModelOutput": [float(x) for x in raw_model_output_flat],
        "modelInputShape": str(model_input_shape),
        "actualInputShape": [list(x_raw_input.shape), list(x_feat_input.shape)],
        "user_id": user_id,
        "debug": {
            "rawMin": raw_min, "rawMax": raw_max,
            "rawMean": raw_mean, "rawStd": raw_std,
        }
    }


def clear_user_model_cache(user_id: str):
    """
    사용자의 모델 캐시를 삭제합니다.
    캘리브레이션 후 새 개인화 모델을 반영하기 위해 사용합니다.
    """
    if user_id in loaded_user_models:
        del loaded_user_models[user_id]
        print(f"🧹 [{user_id}] 모델 캐시 삭제 완료")