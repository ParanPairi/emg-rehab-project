import os
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

from core.config import (
    CNN_MODEL_PATH,
    GESTURE_LABELS,
)
from core.firebase import firestore_db
from services.firebase_service import (
    get_personal_training_count,
    update_personal_model_status,
)
from services.cnn_model_service import loaded_user_models

SCALER_RAW_PATH = "model/scaler_raw_5ch_env.pkl"
SCALER_FEAT_PATH = "model/scaler_feat_5ch_env.pkl"

INITIAL_LR = 0.0005
MIN_LR = 1e-6
BATCH_SIZE = 16
EARLY_STOP_PATIENCE = 3
LR_REDUCE_PATIENCE = 2
MIN_SAMPLES_FOR_VAL_SPLIT = len(GESTURE_LABELS) * 5


REQUIRED_DATA_COUNT = 120
USER_MODEL_DIR = "models/users"


def clear_user_model_cache(user_id: str):
    if user_id in loaded_user_models:
        del loaded_user_models[user_id]


def check_and_train_personal_model(user_id: str):
    """
    /api/predict 이후 백그라운드에서 실행됨.
    개인화 데이터가 충분하면 user_{uid}.h5 생성.
    """

    os.makedirs(USER_MODEL_DIR, exist_ok=True)

    user_model_path = f"{USER_MODEL_DIR}/user_{user_id}.h5"

    if os.path.exists(user_model_path):
        count = get_personal_training_count(user_id)
        update_personal_model_status(
            user_id=user_id,
            mode="personal_model",
            has_personal_model=True,
            training_data_count=count,
            required_data_count=REQUIRED_DATA_COUNT,
            is_training=False,
        )
        return

    count = get_personal_training_count(user_id)

    if count < REQUIRED_DATA_COUNT:
        update_personal_model_status(
            user_id=user_id,
            mode="calibrated_base",
            has_personal_model=False,
            training_data_count=count,
            required_data_count=REQUIRED_DATA_COUNT,
            is_training=False,
        )
        return

    update_personal_model_status(
        user_id=user_id,
        mode="training_personal_model",
        has_personal_model=False,
        training_data_count=count,
        required_data_count=REQUIRED_DATA_COUNT,
        is_training=True,
    )

    try:
        train_user_personal_model(user_id=user_id)

        clear_user_model_cache(user_id)

        update_personal_model_status(
            user_id=user_id,
            mode="personal_model",
            has_personal_model=True,
            training_data_count=count,
            required_data_count=REQUIRED_DATA_COUNT,
            is_training=False,
        )

    except Exception as e:
        print("personal model training error:", e)

        update_personal_model_status(
            user_id=user_id,
            mode="calibrated_base",
            has_personal_model=False,
            training_data_count=count,
            required_data_count=REQUIRED_DATA_COUNT,
            is_training=False,
            error=str(e),
        )


def train_user_personal_model(user_id: str):
    """
    개인화 학습용 데이터로 기본 CNN을 fine-tuning.
    targetGesture가 있는 데이터만 사용.
    캘리브레이션과 동일한 전처리 및 이중 입력 방식 적용.
    """

    docs = (
        firestore_db.collection("users")
        .document(user_id)
        .collection("personalTrainingData")
        .order_by("createdAt")
        .stream()
    )

    x_data = []
    y_data = []
    label_to_idx = {label: idx for idx, label in enumerate(GESTURE_LABELS)}

    for doc in docs:
        item = doc.to_dict()
        target = item.get("targetGesture")
        window_flat = item.get("windowDataFlat")  # ✅ flat 형식으로 변경
        window_length = item.get("windowLength", 25)
        channel_count = item.get("channelCount", 5)

        if target not in label_to_idx:
            continue
        if not isinstance(window_flat, list):
            continue

        try:
            # ✅ flat을 2D로 reshape
            window = np.array(window_flat, dtype=np.float32).reshape(window_length, channel_count)
            if window.ndim != 2 or window.shape[1] != 5:
                continue
            x_data.append(window)
            y_data.append(label_to_idx[target])
        except Exception:
            continue

    if len(x_data) < REQUIRED_DATA_COUNT:
        raise ValueError(
            f"개인화 학습 데이터 부족: {len(x_data)} / {REQUIRED_DATA_COUNT}"
        )

    # 1) 스케일러 로드
    try:
        scaler_raw = joblib.load(SCALER_RAW_PATH)
        scaler_feat = joblib.load(SCALER_FEAT_PATH)
    except Exception as e:
        raise RuntimeError(f"❌ 스케일러 파일을 찾을 수 없습니다: {e}")

    # 2) 캘리브레이션과 동일한 전처리 (베이스라인 조절 + 특징 추출 + 스케일링)
    all_x_adjusted = []
    all_x_feat = []

    for window in x_data:
        baseline = np.min(window, axis=0)
        adj_window = np.clip(window - baseline, a_min=0, a_max=None)

        rms = np.sqrt(np.mean(adj_window ** 2, axis=0))
        mav = np.mean(np.abs(adj_window), axis=0)
        max_val = np.max(adj_window, axis=0)
        var_val = np.var(adj_window, axis=0)
        features = np.concatenate([rms, mav, max_val, var_val])

        all_x_adjusted.append(adj_window)
        all_x_feat.append(features)

    x_raw_arr = np.array(all_x_adjusted, dtype=np.float32)
    x_feat_arr = np.array(all_x_feat, dtype=np.float32)
    y = np.array(y_data, dtype=np.int64)

    x_raw_scaled = scaler_raw.transform(x_raw_arr.reshape(-1, 5)).reshape(x_raw_arr.shape)
    x_feat_scaled = scaler_feat.transform(x_feat_arr)

    # 3) 학습/검증셋 분할
    n_samples = len(y)
    has_val = False

    if n_samples >= MIN_SAMPLES_FOR_VAL_SPLIT:
        try:
            (x_raw_train, x_raw_val,
             x_feat_train, x_feat_val,
             y_train, y_val) = train_test_split(
                x_raw_scaled, x_feat_scaled, y,
                test_size=0.2, stratify=y, random_state=42
            )
            has_val = True
        except Exception:
            x_raw_train, x_feat_train, y_train = x_raw_scaled, x_feat_scaled, y
            x_raw_val, x_feat_val, y_val = np.array([]), np.array([]), np.array([])
    else:
        x_raw_train, x_feat_train, y_train = x_raw_scaled, x_feat_scaled, y
        x_raw_val, x_feat_val, y_val = np.array([]), np.array([]), np.array([])

    # 4) 베이스 모델 로드 + 마지막 4개 레이어만 fine-tuning
    base_model = load_model(CNN_MODEL_PATH, compile=False)

    for layer in base_model.layers[:-4]:
        layer.trainable = False
    for layer in base_model.layers[-4:]:
        layer.trainable = True

    base_model.compile(
        optimizer=Adam(learning_rate=INITIAL_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # 5) 콜백 및 학습
    monitor_metric = "val_loss" if has_val else "loss"
    callbacks = [
        EarlyStopping(monitor=monitor_metric, patience=EARLY_STOP_PATIENCE, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor=monitor_metric, factor=0.5, patience=LR_REDUCE_PATIENCE, min_lr=MIN_LR, verbose=1),
    ]

    fit_kwargs = {
        "x": [x_raw_train, x_feat_train],
        "y": y_train,
        "epochs": 10,
        "batch_size": BATCH_SIZE,
        "verbose": 1,
        "callbacks": callbacks,
        "shuffle": True,
    }
    if has_val:
        fit_kwargs["validation_data"] = ([x_raw_val, x_feat_val], y_val)

    base_model.fit(**fit_kwargs)

    os.makedirs(USER_MODEL_DIR, exist_ok=True)
    save_path = f"{USER_MODEL_DIR}/user_{user_id}.h5"
    base_model.save(save_path)

    print(f"personal model saved: {save_path}")