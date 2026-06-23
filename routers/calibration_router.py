import json
import time
import asyncio
import numpy as np

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from services.calibration_service import run_user_calibration
from core.config import WINDOW_SIZE, CHANNEL_COUNT, GESTURE_LABELS as CONFIG_GESTURE_LABELS

# /ws/emg 로 들어오는 최신 EMG 값을 같이 사용할 수 있게 import
# 라즈베리가 /ws/emg 로만 데이터를 보내도 collect 방식으로 수집 가능
import routers.emg_ws_router as emg_ws_router


router = APIRouter()

# ─── 상수 ───────────────────────────────────────────────────────

# 앱이 window_data를 보내는 방식이면 동작당 여러 개가 쌓임.
# 테스트 단계에서는 너무 높게 잡으면 학습이 안 시작될 수 있어서 1로 둠.
MIN_WINDOWS_PER_GESTURE = 1
MAX_WINDOWS_PER_GESTURE = 80

LOW_VARIANCE_THRESHOLD = 1.0

EPOCHS_MIN = 10
EPOCHS_MAX = 20
EPOCHS_SAMPLES_PER_EPOCH = 50

# collect 방식에서 latest_emg_data를 sliding window로 자를 때 사용
STEP_SIZE = max(1, WINDOW_SIZE // 5)

# ✅ core/config.py와 일치하도록 dict로 변환
GESTURE_LABELS = {name: idx for idx, name in enumerate(CONFIG_GESTURE_LABELS)}


# ─── 헬퍼 함수 ──────────────────────────────────────────────────

def _extract_values_from_latest() -> list[float] | None:
    """
    emg_ws_router.latest_emg_data에서 5채널 values를 꺼냄.
    지원 형식:
      {"values": [...]}
      {"channels": [...]}
      {"ch0": ..., "ch1": ..., "ch2": ..., "ch3": ..., "ch4": ...}
    """
    latest = emg_ws_router.latest_emg_data

    if latest is None:
        return None

    parsed = latest.get("data", {})

    values = parsed.get("values")

    if values is None:
        values = parsed.get("channels")

    if values is None:
        ch_keys = ["ch0", "ch1", "ch2", "ch3", "ch4"]
        if all(k in parsed for k in ch_keys):
            values = [parsed[k] for k in ch_keys]

    if values is None:
        return None

    if not isinstance(values, list):
        return None

    if len(values) < CHANNEL_COUNT:
        return None

    values = values[:CHANNEL_COUNT]

    try:
        return [float(v) for v in values]
    except Exception:
        return None


def _latest_signature() -> str | None:
    """
    같은 latest_emg_data를 반복해서 여러 번 sample_buffer에 넣지 않기 위한 signature.
    time_ms가 있으면 time_ms 기준, 없으면 received_at 기준.
    """
    latest = emg_ws_router.latest_emg_data

    if latest is None:
        return None

    parsed = latest.get("data", {})
    device_id = parsed.get("device_id", "")
    time_ms = parsed.get("time_ms", None)
    t = parsed.get("time", None)

    if time_ms is not None:
        return f"{device_id}:{time_ms}"

    if t is not None:
        return f"{device_id}:{t}"

    return str(latest.get("received_at"))


def _validate_window_array(arr: np.ndarray) -> tuple[bool, str, bool]:
    """
    window_data 검증.
    반환: (accept, error_message, low_variance_warning)
    """
    if arr.shape != (WINDOW_SIZE, CHANNEL_COUNT):
        return False, (
            f"window_data shape 오류: {arr.shape}, "
            f"필요 shape: ({WINDOW_SIZE}, {CHANNEL_COUNT})"
        ), False

    if np.isnan(arr).any():
        return False, "window_data에 NaN 값이 포함되어 있습니다.", False

    if np.isinf(arr).any():
        return False, "window_data에 Inf 값이 포함되어 있습니다.", False

    per_channel_std = np.std(arr, axis=0)
    max_std = float(np.max(per_channel_std))
    low_variance = max_std < LOW_VARIANCE_THRESHOLD

    return True, "", low_variance


def _gesture_counts(y_labels: list[int]) -> dict[str, int]:
    idx_to_name = {v: k for k, v in GESTURE_LABELS.items()}
    counts = {name: 0 for name in GESTURE_LABELS.keys()}

    for label in y_labels:
        name = idx_to_name.get(label)
        if name is not None:
            counts[name] += 1

    return counts


def _flat_y_labels(gesture_buffers: dict[int, list[np.ndarray]]) -> list[int]:
    return [
        idx
        for idx, buf in gesture_buffers.items()
        for _ in buf
    ]


def _check_class_balance(y_labels: list[int]) -> tuple[bool, list[str]]:
    counts = _gesture_counts(y_labels)
    insufficient = [
        name for name, c in counts.items()
        if c < MIN_WINDOWS_PER_GESTURE
    ]
    return len(insufficient) == 0, insufficient


def _decide_epochs(total_samples: int) -> int:
    epochs = EPOCHS_MIN + (total_samples // EPOCHS_SAMPLES_PER_EPOCH)
    return max(EPOCHS_MIN, min(EPOCHS_MAX, epochs))


def _append_window_to_buffer(
    gesture_buffers: dict[int, list[np.ndarray]],
    gesture: str,
    arr: np.ndarray,
) -> tuple[bool, str, bool, int]:
    accept, err_msg, low_variance = _validate_window_array(arr)

    if not accept:
        return False, err_msg, low_variance, 0

    gesture_idx = GESTURE_LABELS[gesture]
    gesture_buffers[gesture_idx].append(arr)

    if len(gesture_buffers[gesture_idx]) > MAX_WINDOWS_PER_GESTURE:
        gesture_buffers[gesture_idx] = gesture_buffers[gesture_idx][-MAX_WINDOWS_PER_GESTURE:]

    return True, "", low_variance, len(gesture_buffers[gesture_idx])


async def _collect_from_latest_emg(
    websocket: WebSocket,
    gesture_buffers: dict[int, list[np.ndarray]],
    gesture: str,
    seconds: float,
):
    """
    앱이 {"type":"collect","gesture":"Rest","seconds":3} 을 보내는 경우 사용.
    /ws/emg에 들어온 latest_emg_data를 seconds 동안 읽어서
    sample_buffer를 만들고 sliding window로 gesture_buffers에 저장.
    """
    if gesture not in GESTURE_LABELS:
        await websocket.send_json({
            "type": "error",
            "message": f"알 수 없는 동작 라벨입니다: {gesture}",
        })
        return

    sample_buffer: list[list[float]] = []
    last_sig = None

    await websocket.send_json({
        "type": "collecting",
        "gesture": gesture,
        "seconds": seconds,
        "message": f"{gesture} 데이터 수집 시작",
    })

    print(f"🎯 [CALIB] {gesture} 수집 시작: {seconds}초")

    start_time = time.time()

    while time.time() - start_time < seconds:
        sig = _latest_signature()

        if sig is not None and sig != last_sig:
            values = _extract_values_from_latest()

            if values is not None and len(values) == CHANNEL_COUNT:
                sample_buffer.append(values)
                last_sig = sig

        await asyncio.sleep(0.005)

    created_windows = 0
    rejected_windows = 0
    low_variance_count = 0

    if len(sample_buffer) >= WINDOW_SIZE:
        for i in range(0, len(sample_buffer) - WINDOW_SIZE + 1, STEP_SIZE):
            window = sample_buffer[i:i + WINDOW_SIZE]
            arr = np.array(window, dtype=np.float32)

            ok, err_msg, low_variance, _ = _append_window_to_buffer(
                gesture_buffers=gesture_buffers,
                gesture=gesture,
                arr=arr,
            )

            if ok:
                created_windows += 1
                if low_variance:
                    low_variance_count += 1
            else:
                rejected_windows += 1
                print(f"⚠️ [CALIB] window 거부: {err_msg}")

    y_labels_flat = _flat_y_labels(gesture_buffers)
    counts = _gesture_counts(y_labels_flat)

    print(
        f"✅ [CALIB] {gesture} 수집 완료 | "
        f"samples={len(sample_buffer)}, windows={created_windows}, counts={counts}"
    )

    await websocket.send_json({
        "type": "collected",
        "gesture": gesture,
        "samples": len(sample_buffer),
        "created_windows": created_windows,
        "rejected_windows": rejected_windows,
        "low_variance_windows": low_variance_count,
        "window_size": WINDOW_SIZE,
        "step_size": STEP_SIZE,
        "counts": counts,
        "total_windows": sum(counts.values()),
        "message": (
            f"{gesture} 데이터 수집 완료: "
            f"샘플 {len(sample_buffer)}개, 윈도우 {created_windows}개 생성"
        ),
    })


# ─── WebSocket 엔드포인트 ─────────────────────────────────────────

@router.websocket("/ws/calibration/{user_id}")
async def calibration_websocket(websocket: WebSocket, user_id: str):
    await websocket.accept()
    print(f"🔌 [{user_id}] 캘리브레이션 WebSocket 연결됨")

    gesture_buffers: dict[int, list[np.ndarray]] = {
        idx: [] for idx in GESTURE_LABELS.values()
    }

    try:
        await websocket.send_json({
            "type": "connected",
            "message": "캘리브레이션 WebSocket 연결 완료",
            "user_id": user_id,
            "config": {
                "window_size": WINDOW_SIZE,
                "channel_count": CHANNEL_COUNT,
                "min_per_gesture": MIN_WINDOWS_PER_GESTURE,
                "max_per_gesture": MAX_WINDOWS_PER_GESTURE,
                "step_size": STEP_SIZE,
                "gestures": list(GESTURE_LABELS.keys()),
            },
        })

        while True:
            raw_message = await websocket.receive_text()

            try:
                data = json.loads(raw_message)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "JSON 형식이 올바르지 않습니다.",
                })
                continue

            message_type = data.get("type")
            print(f"📩 [CALIB] 받은 메시지: {data if message_type != 'window' else {'type': data.get('type'), 'gesture': data.get('gesture'), 'window_len': len(data.get('window_data', [])) if isinstance(data.get('window_data'), list) else None}}")

            # ─── 1. 전체 초기화 ───────────────────────────────────
            if message_type == "reset":
                for idx in gesture_buffers:
                    gesture_buffers[idx].clear()

                await websocket.send_json({
                    "type": "reset_done",
                    "message": "캘리브레이션 데이터를 초기화했습니다.",
                    "counts": _gesture_counts([]),
                })

            # ─── 2. start ────────────────────────────────────────
            elif message_type == "start":
                gesture = data.get("gesture")
                seconds = float(data.get("seconds", 3.0))

                if gesture is None:
                    for idx in gesture_buffers:
                        gesture_buffers[idx].clear()

                    await websocket.send_json({
                        "type": "started",
                        "message": "캘리브레이션 데이터 수집을 시작합니다.",
                        "counts": _gesture_counts([]),
                    })
                else:
                    await _collect_from_latest_emg(
                        websocket=websocket,
                        gesture_buffers=gesture_buffers,
                        gesture=gesture,
                        seconds=seconds,
                    )

            # ─── 3. collect 방식 지원 ─────────────────────────────
            elif message_type in ("collect", "measure", "gesture_collect"):
                gesture = data.get("gesture")
                seconds = float(data.get("seconds", 3.0))

                await _collect_from_latest_emg(
                    websocket=websocket,
                    gesture_buffers=gesture_buffers,
                    gesture=gesture,
                    seconds=seconds,
                )

            # ─── 4. 앱이 window_data를 직접 보내는 기존 방식 유지 ───
            elif message_type == "window":
                gesture = data.get("gesture")
                window_data = data.get("window_data")

                if gesture not in GESTURE_LABELS:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"알 수 없는 동작 라벨입니다: {gesture}",
                    })
                    continue

                if not isinstance(window_data, list):
                    await websocket.send_json({
                        "type": "error",
                        "message": "window_data가 list 형식이 아닙니다.",
                    })
                    continue

                try:
                    arr = np.array(window_data, dtype=np.float32)
                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"window_data 변환 실패: {e}",
                    })
                    continue

                ok, err_msg, low_variance, current_count = _append_window_to_buffer(
                    gesture_buffers=gesture_buffers,
                    gesture=gesture,
                    arr=arr,
                )

                if not ok:
                    await websocket.send_json({
                        "type": "error",
                        "message": err_msg,
                    })
                    continue

                y_labels_flat = _flat_y_labels(gesture_buffers)
                counts = _gesture_counts(y_labels_flat)

                response = {
                    "type": "received",
                    "gesture": gesture,
                    "gesture_count": current_count,
                    "saved_windows": current_count,
                    "gesture_min": MIN_WINDOWS_PER_GESTURE,
                    "gesture_max": MAX_WINDOWS_PER_GESTURE,
                    "gesture_full": current_count >= MAX_WINDOWS_PER_GESTURE,
                    "gesture_min_met": current_count >= MIN_WINDOWS_PER_GESTURE,
                    "total_windows": sum(counts.values()),
                    "counts": counts,
                }

                if low_variance:
                    response["warning"] = (
                        "신호 분산이 매우 낮습니다. 센서 부착 상태나 "
                        "동작 수행을 확인해주세요. (학습에는 포함됨)"
                    )

                print(f"✅ [CALIB] window 저장 완료: gesture={gesture}, counts={counts}")
                await websocket.send_json(response)

            # ─── 5. 진행상황 조회 ─────────────────────────────────
            elif message_type == "status":
                y_labels_flat = _flat_y_labels(gesture_buffers)
                counts = _gesture_counts(y_labels_flat)
                balanced, insufficient = _check_class_balance(y_labels_flat)

                await websocket.send_json({
                    "type": "status",
                    "total_windows": sum(counts.values()),
                    "counts": counts,
                    "all_min_met": balanced,
                    "insufficient_gestures": insufficient,
                    "min_per_gesture": MIN_WINDOWS_PER_GESTURE,
                })

            # ─── 6. 학습 시작 ─────────────────────────────────────
            elif message_type == "finish":
                x_windows = []
                y_labels = []

                for idx, buf in gesture_buffers.items():
                    for w in buf:
                        x_windows.append(w)
                        y_labels.append(idx)

                counts = _gesture_counts(y_labels)

                print(f"🏁 [CALIB] finish 요청")
                print(f"🏁 [CALIB] total_windows={len(x_windows)}")
                print(f"🏁 [CALIB] counts={counts}")

                if len(x_windows) == 0:
                    await websocket.send_json({
                        "type": "error",
                        "message": (
                            "학습할 캘리브레이션 데이터가 없습니다. "
                            "각 동작 window가 서버로 전송되는지 확인해야 합니다."
                        ),
                        "counts": counts,
                    })
                    continue

                balanced, insufficient = _check_class_balance(y_labels)

                if not balanced:
                    await websocket.send_json({
                        "type": "insufficient_data",
                        "message": (
                            f"학습을 시작하려면 동작당 최소 "
                            f"{MIN_WINDOWS_PER_GESTURE}개의 윈도우가 필요합니다."
                        ),
                        "counts": counts,
                        "insufficient_gestures": insufficient,
                        "min_per_gesture": MIN_WINDOWS_PER_GESTURE,
                    })
                    continue

                total = len(x_windows)
                epochs = _decide_epochs(total)

                await websocket.send_json({
                    "type": "training_started",
                    "message": "사용자 맞춤 모델 학습을 시작합니다.",
                    "total_windows": total,
                    "counts": counts,
                    "epochs": epochs,
                })

                x_calib_raw = np.stack(x_windows, axis=0)
                y_calib = np.array(y_labels, dtype=np.int64)

                print(f"🚀 [CALIB] x_calib_raw shape={x_calib_raw.shape}")
                print(f"🚀 [CALIB] y_calib shape={y_calib.shape}")
                print(f"🚀 [CALIB] unique labels={np.unique(y_calib, return_counts=True)}")

                try:
                    result = await run_in_threadpool(
                        run_user_calibration,
                        user_id=user_id,
                        x_calib_raw=x_calib_raw,
                        y_calib=y_calib,
                        epochs=epochs,
                    )
                except Exception as e:
                    print(f"🔥 [{user_id}] 학습 중 오류: {e}")
                    await websocket.send_json({
                        "type": "training_failed",
                        "message": f"학습 중 오류가 발생했습니다: {e}",
                    })
                    continue

                await websocket.send_json({
                    "type": "complete",
                    "message": "캘리브레이션 및 사용자 모델 학습 완료",
                    "result": result,
                })

            # ─── 7. 종료 ─────────────────────────────────────────
            elif message_type == "close":
                await websocket.send_json({
                    "type": "closed",
                    "message": "캘리브레이션 연결을 종료합니다.",
                })
                await websocket.close()
                break

            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"알 수 없는 message type입니다: {message_type}",
                })

    except WebSocketDisconnect:
        print(f"❌ [{user_id}] 캘리브레이션 WebSocket 연결 끊김")

    except Exception as e:
        print(f"🔥 [{user_id}] 캘리브레이션 WebSocket 오류: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e),
            })
        except Exception:
            pass
