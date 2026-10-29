import numpy as np

from core.config import WINDOW_SIZE, CHANNEL_COUNT


def validate_emg_window(window_data: list[list[float]]) -> np.ndarray:
    raw_window = np.array(window_data, dtype=float)

    if raw_window.ndim != 2 or raw_window.shape[1] != CHANNEL_COUNT:
        raise ValueError(
            f"window_data는 Nx{CHANNEL_COUNT} 형태의 2차원 배열이어야 합니다. "
            f"received_shape={list(raw_window.shape)}"
        )

    if raw_window.shape[0] < WINDOW_SIZE:
        raise ValueError(
            f"EMG 데이터가 {WINDOW_SIZE}개 미만입니다. "
            f"received_count={int(raw_window.shape[0])}"
        )

    if np.isnan(raw_window).any():
        raise ValueError("window_data에 NaN 값이 포함되어 있습니다.")

    if np.isinf(raw_window).any():
        raise ValueError("window_data에 Inf 값이 포함되어 있습니다.")

    return raw_window[-WINDOW_SIZE:]


def extract_features(raw_window: np.ndarray) -> dict:
    rms = np.sqrt(np.mean(raw_window ** 2, axis=0))
    mav = np.mean(np.abs(raw_window), axis=0)
    wl = np.sum(np.abs(np.diff(raw_window, axis=0)), axis=0)
    
    # ✅ 캘리브레이션과 일치하도록 MAX, VAR 추가
    max_val = np.max(raw_window, axis=0)
    var_val = np.var(raw_window, axis=0)
    
    # (20,) 차원으로 통일: [RMS, MAV, MAX, VAR] = 5채널 × 4 = 20
    features = np.concatenate([rms, mav, max_val, var_val])

    return {
        "rms": rms.tolist(),  # ✅ numpy array → list
        "mav": mav.tolist(),  # ✅ numpy array → list
        "wl": wl.tolist(),    # ✅ numpy array → list
        "max": max_val.tolist(),  # ✅ numpy array → list
        "var": var_val.tolist(),  # ✅ numpy array → list
        "features": features.tolist(),  # ✅ numpy array → list
        "rms_avg": float(np.mean(rms)),
        "mav_avg": float(np.mean(mav)),
        "wl_avg": float(np.mean(wl)),
    }