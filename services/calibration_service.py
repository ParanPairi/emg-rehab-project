import os
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

from firebase_admin import storage, firestore

from core.config import CNN_MODEL_PATH
from core.firebase import firestore_db
from services.cnn_model_service import clear_user_model_cache

USER_MODELS_DIR = "models/users"
os.makedirs(USER_MODELS_DIR, exist_ok=True)

GESTURE_LABELS = [
    "Rest",
    "Fist",
    "Wrist Flexion",
    "Wrist Extension",
    "Radial Deviation",
    "Ulnar Deviation",
]

VAL_SPLIT_RATIO = 0.2
EARLY_STOP_PATIENCE = 3
LR_REDUCE_PATIENCE = 2
INITIAL_LR = 0.0005
MIN_LR = 1e-6
BATCH_SIZE = 16
MIN_SAMPLES_FOR_VAL_SPLIT = len(GESTURE_LABELS) * 5

SCALER_RAW_PATH = "model/scaler_raw_5ch_env.pkl"
SCALER_FEAT_PATH = "model/scaler_feat_5ch_env.pkl"


def upload_model_to_firebase(user_id: str, local_path: str) -> str:
    """학습된 개인화 모델 파일을 Firebase Storage에 업로드"""
    try:
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"로컬 모델 파일 미발견: {local_path}")
        
        storage_path = f"models/users/user_{user_id}.h5"
        bucket = storage.bucket()
        blob = bucket.blob(storage_path)
        blob.upload_from_filename(local_path)
        print(f"☁️ [{user_id}] Firebase Storage 업로드 완료: {storage_path}")
        return storage_path
    except Exception as e:
        print(f"⚠️ Firebase 업로드 실패: {e}")
        raise


def save_personalized_model_info(
    user_id: str,
    local_model_path: str,
    storage_model_path: str,
    train_samples: int,
    val_samples: int,
    epochs_requested: int,
    epochs_ran: int,
    train_accuracy: float | None = None,
    val_accuracy: float | None = None,
    val_loss: float | None = None,
    upload_success: bool = True,
):
    """개인화 모델 정보를 Firestore에 저장"""
    model_id = "latest"

    model_data = {
        "modelId": model_id,
        "modelType": "cnn_finetuned",
        "baseModelPath": CNN_MODEL_PATH,
        "localModelPath": local_model_path,
        "storageModelPath": storage_model_path,
        "trainSamples": train_samples,
        "valSamples": val_samples,
        "epochsRequested": epochs_requested,
        "epochsRan": epochs_ran,
        "gestureLabels": GESTURE_LABELS,
        "isActive": upload_success,
        "uploadSuccess": upload_success,
        "createdAt": firestore.SERVER_TIMESTAMP,
        "updatedAt": firestore.SERVER_TIMESTAMP,
    }

    if train_accuracy is not None:
        model_data["trainAccuracy"] = train_accuracy
    if val_accuracy is not None:
        model_data["valAccuracy"] = val_accuracy
        model_data["accuracy"] = val_accuracy
    if val_loss is not None:
        model_data["valLoss"] = val_loss

    firestore_db.collection("users").document(user_id).collection("personalizedModels").document(model_id).set(model_data, merge=True)
    firestore_db.collection("users").document(user_id).set({
        "calibrationCompleted": True,
        "personalizedModelAvailable": upload_success,
        "activePersonalizedModelId": model_id,
        "lastCalibrationAt": firestore.SERVER_TIMESTAMP,
    }, merge=True)

    print(f"🔥 [{user_id}] 개인화 모델 정보 Firestore 저장 완료 (upload_success={upload_success})")


def _validate_calibration_input(
    x_calib_raw: np.ndarray,
    y_calib: np.ndarray,
) -> None:
    """캘리브레이션 입력 데이터 검증"""
    if x_calib_raw is None or y_calib is None:
        raise ValueError("x_calib_raw 또는 y_calib가 None입니다.")

    if len(x_calib_raw) == 0:
        raise ValueError("캘리브레이션 데이터가 비어 있습니다.")

    if len(x_calib_raw) != len(y_calib):
        raise ValueError(f"x_calib_raw와 y_calib의 샘플 수가 다릅니다: {len(x_calib_raw)} vs {len(y_calib)}")

    unique_labels = np.unique(y_calib)
    if len(unique_labels) < 2:
        raise ValueError(f"캘리브레이션 데이터의 클래스가 부족합니다 (최소 2개 필요): found={unique_labels.tolist()}")

    invalid_labels = [int(lbl) for lbl in unique_labels if lbl < 0 or lbl >= len(GESTURE_LABELS)]
    if invalid_labels:
        raise ValueError(f"잘못된 라벨이 포함되어 있습니다: {invalid_labels} (허용 범위: 0 ~ {len(GESTURE_LABELS) - 1})")

def run_user_calibration(
    user_id: str,
    x_calib_raw: np.ndarray,
    y_calib: np.ndarray,
    epochs: int = 10,
) -> dict:
    print(f"🚀 [{user_id}] ENV 하이브리드 캘리브레이션 시작...")

    # 1) 입력 검증
    _validate_calibration_input(x_calib_raw, y_calib)

    # 2) 스케일러 로드 (반드시 학습 때 저장한 2개의 스케일러를 불러와야 합니다)
    try:
        if not os.path.exists(SCALER_RAW_PATH):
            raise FileNotFoundError(f"스케일러 파일 미발견: {SCALER_RAW_PATH}")
        if not os.path.exists(SCALER_FEAT_PATH):
            raise FileNotFoundError(f"스케일러 파일 미발견: {SCALER_FEAT_PATH}")
        
        scaler_raw = joblib.load(SCALER_RAW_PATH)
        scaler_feat = joblib.load(SCALER_FEAT_PATH)
        print(f"📦 스케일러 로드 완료")
    except Exception as e:
        raise RuntimeError(f"❌ 스케일러 파일을 찾을 수 없습니다: {e}")

    # 3) ENV 전처리 및 특징 추출 (다중 입력 생성)
    all_x_adjusted = []
    all_x_feat = []

    # x_calib_raw 형태가 (윈도우 수, 25, 5) 라고 가정
    for window in x_calib_raw:
        # 베이스라인 영점 조절 (최솟값을 0으로)
        baseline = np.min(window, axis=0)
        adj_window = np.clip(window - baseline, a_min=0, a_max=None)
        
        # 특징 4종 추출
        rms = np.sqrt(np.mean(adj_window**2, axis=0))
        mav = np.mean(np.abs(adj_window), axis=0)
        max_val = np.max(adj_window, axis=0)
        var_val = np.var(adj_window, axis=0)
        
        features = np.concatenate([rms, mav, max_val, var_val])
        
        all_x_adjusted.append(adj_window)
        all_x_feat.append(features)

    x_raw_arr = np.array(all_x_adjusted, dtype=np.float32)
    x_feat_arr = np.array(all_x_feat, dtype=np.float32)
    y_calib = y_calib.astype("int64")

    # 4) 스케일링 적용 (transform)
    x_raw_reshaped = x_raw_arr.reshape(-1, 5) # 5채널
    x_raw_scaled = scaler_raw.transform(x_raw_reshaped).reshape(x_raw_arr.shape)
    x_feat_scaled = scaler_feat.transform(x_feat_arr)

    print(f"📦 변환 완료: x_raw={x_raw_scaled.shape}, x_feat={x_feat_scaled.shape}")

    # 5) 학습/검증셋 분할 (다중 입력이므로 직접 train_test_split 사용)
    n_samples = len(y_calib)
    has_val = False

    if n_samples < MIN_SAMPLES_FOR_VAL_SPLIT:
        print(f"⚠️ 샘플 수가 부족해 검증셋 분할 생략")
        x_raw_train, x_feat_train, y_train = x_raw_scaled, x_feat_scaled, y_calib
        x_raw_val, x_feat_val, y_val = np.array([]), np.array([]), np.array([])
    else:
        try:
            (x_raw_train, x_raw_val, 
             x_feat_train, x_feat_val, 
             y_train, y_val) = train_test_split(
                x_raw_scaled, x_feat_scaled, y_calib, 
                test_size=VAL_SPLIT_RATIO, stratify=y_calib, random_state=42
            )
            has_val = True
        except Exception:
            print(f"⚠️ Stratified 분할 실패. 전체를 학습에 사용합니다.")
            x_raw_train, x_feat_train, y_train = x_raw_scaled, x_feat_scaled, y_calib
            x_raw_val, x_feat_val, y_val = np.array([]), np.array([]), np.array([])

    # 6) 베이스 모델 로드 + fine-tuning 준비
    try:
        if not os.path.exists(CNN_MODEL_PATH):
            raise FileNotFoundError(f"모델 파일 미발견: {CNN_MODEL_PATH}")
        base_model = load_model(CNN_MODEL_PATH)
        print(f"📦 베이스 모델 로드 완료: {CNN_MODEL_PATH}")
    except Exception as e:
        raise RuntimeError(f"❌ 베이스 모델을 로드할 수 없습니다: {e}")

    # 하이브리드 모델은 레이어 구조가 다르므로 뒤쪽 Dense 위주로 풀어줍니다
    # (일반적으로 마지막 3~4개 레이어가 Classifier 역할을 합니다)
    for layer in base_model.layers[:-4]:
        layer.trainable = False
    for layer in base_model.layers[-4:]:
        layer.trainable = True

    base_model.compile(
        optimizer=Adam(learning_rate=INITIAL_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # 7) 콜백 구성
    callbacks = []
    monitor_metric = "val_loss" if has_val else "loss"

    callbacks.append(EarlyStopping(monitor=monitor_metric, patience=EARLY_STOP_PATIENCE, restore_best_weights=True, verbose=1))
    callbacks.append(ReduceLROnPlateau(monitor=monitor_metric, factor=0.5, patience=LR_REDUCE_PATIENCE, min_lr=MIN_LR, verbose=1))

    # 8) 학습 (다중 입력 리스트 전달 ⭐️ 가장 중요)
    fit_kwargs = {
        "x": [x_raw_train, x_feat_train],  # <-- 리스트 형태로 두 개의 데이터를 줍니다!
        "y": y_train,
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "verbose": 1,
        "callbacks": callbacks,
        "shuffle": True,
    }
    if has_val:
        fit_kwargs["validation_data"] = ([x_raw_val, x_feat_val], y_val)

    history = base_model.fit(**fit_kwargs)

    # 9) 결과 메트릭 추출 및 저장 (기존 7~10번 로직 동일하게 유지)
    train_accuracy = float(history.history["accuracy"][-1]) if "accuracy" in history.history else None
    val_accuracy = float(history.history["val_accuracy"][-1]) if has_val and "val_accuracy" in history.history else None
    val_loss = float(history.history["val_loss"][-1]) if has_val and "val_loss" in history.history else None
    epochs_ran = len(history.history["accuracy"]) if "accuracy" in history.history else 0

    local_path = os.path.join(USER_MODELS_DIR, f"user_{user_id}.h5")
    base_model.save(local_path)
    clear_user_model_cache(user_id)

    # Firebase 업로드 및 메타데이터 저장 로직
    storage_path = ""
    upload_success = False
    try:
        storage_path = upload_model_to_firebase(user_id, local_path)
        upload_success = True
    except Exception as e:
        print(f"❌ Firebase Storage 업로드 실패: {e}")

    try:
        save_personalized_model_info(
            user_id=user_id, local_model_path=local_path, storage_model_path=storage_path,
            train_samples=len(x_raw_train), val_samples=len(x_raw_val), epochs_requested=epochs, epochs_ran=epochs_ran,
            train_accuracy=train_accuracy, val_accuracy=val_accuracy, val_loss=val_loss, upload_success=upload_success,
        )
    except Exception as e:
        print(f"❌ Firestore 메타데이터 저장 실패: {e}")

    return {
        "status": "success",
        "user_id": user_id,
        "model_id": "latest",
        "model_path": local_path,
        "upload_success": upload_success,
        "epochs_ran": epochs_ran,
        "accuracy": val_accuracy if val_accuracy is not None else train_accuracy,
    }