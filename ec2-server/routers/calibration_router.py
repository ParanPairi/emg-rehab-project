import json
import numpy as np

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from services.calibration_service import run_user_calibration
from core.config import (
    WINDOW_SIZE,
    CHANNEL_COUNT,
    GESTURE_LABELS as CONFIG_GESTURE_LABELS,
)

router = APIRouter()

MIN_WINDOWS_PER_GESTURE = 1
MAX_WINDOWS_PER_GESTURE = 80

LOW_VARIANCE_THRESHOLD = 1.0

EPOCHS_MIN = 10
EPOCHS_MAX = 20
EPOCHS_SAMPLES_PER_EPOCH = 50

GESTURE_LABELS = {
    name: idx for idx, name in enumerate(CONFIG_GESTURE_LABELS)
}

# 같은 user_id의 앱과 라즈베리파이가 하나의 세션을 공유한다.
calibration_sessions: dict[str, dict] = {}


def _new_gesture_buffers() -> dict[int, list[np.ndarray]]:
    return {idx: [] for idx in GESTURE_LABELS.values()}


def _get_or_create_session(user_id: str) -> dict:
    session = calibration_sessions.get(user_id)

    if session is None:
        session = {
            "buffers": _new_gesture_buffers(),
            "app_clients": set(),
            "device_client": None,
            "state": "idle",
            "last_progress": None,
        }
        calibration_sessions[user_id] = session

    return session


def _clear_session_buffers(session: dict) -> None:
    session["buffers"] = _new_gesture_buffers()


async def _broadcast_to_apps(user_id: str, message: dict) -> None:
    session = _get_or_create_session(user_id)
    disconnected = []

    for client in list(session["app_clients"]):
        try:
            await client.send_json(message)
        except Exception:
            disconnected.append(client)

    for client in disconnected:
        session["app_clients"].discard(client)


async def _send_to_device_and_apps(
    websocket: WebSocket,
    user_id: str,
    message: dict,
) -> None:
    # 라즈베리파이는 received 응답을 기다리므로 원래 연결에도 응답한다.
    await websocket.send_json(message)

    # 같은 사용자로 연결된 앱에도 진행상황을 전달한다.
    await _broadcast_to_apps(user_id, message)


def _validate_window_array(
    arr: np.ndarray,
) -> tuple[bool, str, bool]:
    if arr.shape != (WINDOW_SIZE, CHANNEL_COUNT):
        return (
            False,
            (
                f"window_data shape 오류: {arr.shape}, "
                f"필요 shape: ({WINDOW_SIZE}, {CHANNEL_COUNT})"
            ),
            False,
        )

    if np.isnan(arr).any():
        return False, "window_data에 NaN 값이 포함되어 있습니다.", False

    if np.isinf(arr).any():
        return False, "window_data에 Inf 값이 포함되어 있습니다.", False

    per_channel_std = np.std(arr, axis=0)
    max_std = float(np.max(per_channel_std))
    low_variance = max_std < LOW_VARIANCE_THRESHOLD

    return True, "", low_variance


def _gesture_counts(y_labels: list[int]) -> dict[str, int]:
    idx_to_name = {value: key for key, value in GESTURE_LABELS.items()}
    counts = {name: 0 for name in GESTURE_LABELS.keys()}

    for label in y_labels:
        name = idx_to_name.get(label)
        if name is not None:
            counts[name] += 1

    return counts


def _flat_y_labels(
    gesture_buffers: dict[int, list[np.ndarray]],
) -> list[int]:
    return [
        idx
        for idx, buffer in gesture_buffers.items()
        for _ in buffer
    ]


def _check_class_balance(
    y_labels: list[int],
) -> tuple[bool, list[str]]:
    counts = _gesture_counts(y_labels)
    insufficient = [
        name
        for name, count in counts.items()
        if count < MIN_WINDOWS_PER_GESTURE
    ]
    return len(insufficient) == 0, insufficient


def _decide_epochs(total_samples: int) -> int:
    epochs = EPOCHS_MIN + (
        total_samples // EPOCHS_SAMPLES_PER_EPOCH
    )
    return max(EPOCHS_MIN, min(EPOCHS_MAX, epochs))


def _append_window_to_buffer(
    gesture_buffers: dict[int, list[np.ndarray]],
    gesture: str,
    arr: np.ndarray,
) -> tuple[bool, str, bool, int]:
    accepted, error_message, low_variance = (
        _validate_window_array(arr)
    )

    if not accepted:
        return False, error_message, low_variance, 0

    gesture_idx = GESTURE_LABELS[gesture]
    gesture_buffers[gesture_idx].append(arr)

    if len(gesture_buffers[gesture_idx]) > MAX_WINDOWS_PER_GESTURE:
        gesture_buffers[gesture_idx] = (
            gesture_buffers[gesture_idx][-MAX_WINDOWS_PER_GESTURE:]
        )

    return (
        True,
        "",
        low_variance,
        len(gesture_buffers[gesture_idx]),
    )


@router.websocket("/ws/calibration/{user_id}")
async def calibration_websocket(
    websocket: WebSocket,
    user_id: str,
):
    # 앱:    ?role=app
    # 라즈베리파이: ?role=device
    role = (
        websocket.query_params.get("role", "app")
        .strip()
        .lower()
    )
    if role not in {"app", "device"}:
        role = "app"

    await websocket.accept()
    session = _get_or_create_session(user_id)

    if role == "device":
        previous_device = session.get("device_client")

        if (
            previous_device is not None
            and previous_device is not websocket
        ):
            try:
                await previous_device.close()
            except Exception:
                pass

        session["device_client"] = websocket
    else:
        session["app_clients"].add(websocket)

    print(
        f"🔌 [{user_id}] 캘리브레이션 WebSocket 연결됨 "
        f"role={role}"
    )
    print(
        f"⚙️ [CALIB] WINDOW_SIZE={WINDOW_SIZE}, "
        f"CHANNEL_COUNT={CHANNEL_COUNT}"
    )

    labels = _flat_y_labels(session["buffers"])
    counts = _gesture_counts(labels)

    await websocket.send_json({
        "type": "connected",
        "message": "캘리브레이션 WebSocket 연결 완료",
        "user_id": user_id,
        "role": role,
        "state": session["state"],
        "counts": counts,
        "config": {
            "window_size": WINDOW_SIZE,
            "channel_count": CHANNEL_COUNT,
            "min_per_gesture": MIN_WINDOWS_PER_GESTURE,
            "max_per_gesture": MAX_WINDOWS_PER_GESTURE,
            "gestures": list(GESTURE_LABELS.keys()),
            "mode": "device_window",
        },
    })

    # 앱이 늦게 연결되었을 경우 마지막 진행상황을 한 번 다시 보낸다.
    if role == "app" and session["last_progress"] is not None:
        try:
            await websocket.send_json(session["last_progress"])
        except Exception:
            pass

    try:
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

            # 앱 연결은 진행상황 수신 전용이다.
            if role == "app":
                if message_type == "status":
                    labels = _flat_y_labels(session["buffers"])
                    counts = _gesture_counts(labels)
                    balanced, insufficient = (
                        _check_class_balance(labels)
                    )

                    await websocket.send_json({
                        "type": "status",
                        "state": session["state"],
                        "total_windows": sum(counts.values()),
                        "counts": counts,
                        "all_min_met": balanced,
                        "insufficient_gestures": insufficient,
                        "min_per_gesture": MIN_WINDOWS_PER_GESTURE,
                    })
                    continue

                if message_type == "close":
                    await websocket.send_json({
                        "type": "closed",
                        "message": "앱 연결을 종료합니다.",
                    })
                    await websocket.close()
                    break

                await websocket.send_json({
                    "type": "error",
                    "message": "앱 연결은 진행상황 수신 전용입니다.",
                })
                continue

            # 아래부터는 role=device인 라즈베리파이 메시지 처리
            if message_type == "window":
                window_data = data.get("window_data")
                window_len = (
                    len(window_data)
                    if isinstance(window_data, list)
                    else None
                )
                print(
                    "📩 [CALIB] 받은 메시지: "
                    f"{{'type': 'window', "
                    f"'gesture': {data.get('gesture')}, "
                    f"'window_len': {window_len}}}"
                )
            else:
                print(f"📩 [CALIB] 받은 메시지: {data}")

            if message_type == "progress":
                progress = {
                    "type": "progress",
                    "gesture": data.get("gesture"),
                    "phase": data.get("phase"),
                    "remaining_seconds": data.get(
                        "remaining_seconds"
                    ),
                    "step_index": data.get("step_index"),
                    "total_steps": data.get(
                        "total_steps",
                        len(GESTURE_LABELS),
                    ),
                    "message": data.get("message"),
                }

                session["state"] = (
                    data.get("phase") or "collecting"
                )
                session["last_progress"] = progress

                await _broadcast_to_apps(user_id, progress)
                continue

            if message_type == "reset":
                _clear_session_buffers(session)
                session["state"] = "idle"
                session["last_progress"] = None

                response = {
                    "type": "reset_done",
                    "message": "캘리브레이션 데이터를 초기화했습니다.",
                    "counts": _gesture_counts([]),
                }
                await _send_to_device_and_apps(
                    websocket,
                    user_id,
                    response,
                )

            elif message_type == "start":
                _clear_session_buffers(session)
                session["state"] = "collecting"
                session["last_progress"] = None

                response = {
                    "type": "started",
                    "message": (
                        "라즈베리파이에서 캘리브레이션 "
                        "데이터 수집을 시작합니다."
                    ),
                    "counts": _gesture_counts([]),
                }
                await _send_to_device_and_apps(
                    websocket,
                    user_id,
                    response,
                )

            elif message_type == "window":
                gesture = data.get("gesture")
                window_data = data.get("window_data")

                if gesture not in GESTURE_LABELS:
                    response = {
                        "type": "error",
                        "message": (
                            f"알 수 없는 동작 라벨입니다: {gesture}"
                        ),
                    }
                    await _send_to_device_and_apps(
                        websocket,
                        user_id,
                        response,
                    )
                    continue

                if not isinstance(window_data, list):
                    response = {
                        "type": "error",
                        "message": (
                            "window_data가 list 형식이 아닙니다."
                        ),
                    }
                    await _send_to_device_and_apps(
                        websocket,
                        user_id,
                        response,
                    )
                    continue

                try:
                    arr = np.array(
                        window_data,
                        dtype=np.float32,
                    )
                except Exception as error:
                    response = {
                        "type": "error",
                        "message": (
                            f"window_data 변환 실패: {error}"
                        ),
                    }
                    await _send_to_device_and_apps(
                        websocket,
                        user_id,
                        response,
                    )
                    continue

                (
                    ok,
                    error_message,
                    low_variance,
                    current_count,
                ) = _append_window_to_buffer(
                    gesture_buffers=session["buffers"],
                    gesture=gesture,
                    arr=arr,
                )

                if not ok:
                    print(
                        "⚠️ [CALIB] window 거부: "
                        f"gesture={gesture}, "
                        f"error={error_message}"
                    )
                    response = {
                        "type": "error",
                        "message": error_message,
                    }
                    await _send_to_device_and_apps(
                        websocket,
                        user_id,
                        response,
                    )
                    continue

                labels = _flat_y_labels(session["buffers"])
                counts = _gesture_counts(labels)

                response = {
                    "type": "received",
                    "gesture": gesture,
                    "gesture_count": current_count,
                    "saved_windows": current_count,
                    "gesture_min": MIN_WINDOWS_PER_GESTURE,
                    "gesture_max": MAX_WINDOWS_PER_GESTURE,
                    "gesture_full": (
                        current_count >= MAX_WINDOWS_PER_GESTURE
                    ),
                    "gesture_min_met": (
                        current_count >= MIN_WINDOWS_PER_GESTURE
                    ),
                    "total_windows": sum(counts.values()),
                    "counts": counts,
                }

                if low_variance:
                    response["warning"] = (
                        "신호 분산이 매우 낮습니다. "
                        "센서 부착 상태나 동작 수행을 확인해주세요. "
                        "학습에는 포함했습니다."
                    )

                print(
                    "✅ [CALIB] window 저장 완료: "
                    f"gesture={gesture}, counts={counts}"
                )

                await _send_to_device_and_apps(
                    websocket,
                    user_id,
                    response,
                )

            elif message_type == "gesture_complete":
                response = {
                    "type": "gesture_complete",
                    "gesture": data.get("gesture"),
                    "windows_sent": data.get(
                        "windows_sent",
                        0,
                    ),
                    "step_index": data.get("step_index"),
                    "total_steps": data.get(
                        "total_steps",
                        len(GESTURE_LABELS),
                    ),
                    "message": data.get("message"),
                }
                await _broadcast_to_apps(user_id, response)

            elif message_type == "status":
                labels = _flat_y_labels(session["buffers"])
                counts = _gesture_counts(labels)
                balanced, insufficient = (
                    _check_class_balance(labels)
                )

                await websocket.send_json({
                    "type": "status",
                    "state": session["state"],
                    "total_windows": sum(counts.values()),
                    "counts": counts,
                    "all_min_met": balanced,
                    "insufficient_gestures": insufficient,
                    "min_per_gesture": MIN_WINDOWS_PER_GESTURE,
                })

            elif message_type == "finish":
                x_windows = []
                y_labels = []

                for idx, buffer in session["buffers"].items():
                    for window in buffer:
                        x_windows.append(window)
                        y_labels.append(idx)

                counts = _gesture_counts(y_labels)

                print("🏁 [CALIB] finish 요청")
                print(
                    f"🏁 [CALIB] total_windows={len(x_windows)}"
                )
                print(f"🏁 [CALIB] counts={counts}")

                if len(x_windows) == 0:
                    response = {
                        "type": "error",
                        "message": (
                            "학습할 캘리브레이션 데이터가 없습니다."
                        ),
                        "counts": counts,
                    }
                    await _send_to_device_and_apps(
                        websocket,
                        user_id,
                        response,
                    )
                    continue

                balanced, insufficient = (
                    _check_class_balance(y_labels)
                )

                if not balanced:
                    response = {
                        "type": "insufficient_data",
                        "message": (
                            "학습을 시작하려면 동작당 최소 "
                            f"{MIN_WINDOWS_PER_GESTURE}개의 "
                            "윈도우가 필요합니다."
                        ),
                        "counts": counts,
                        "insufficient_gestures": insufficient,
                        "min_per_gesture": MIN_WINDOWS_PER_GESTURE,
                    }
                    await _send_to_device_and_apps(
                        websocket,
                        user_id,
                        response,
                    )
                    continue

                total = len(x_windows)
                epochs = _decide_epochs(total)
                session["state"] = "training"

                training_message = {
                    "type": "training_started",
                    "message": "사용자 맞춤 모델 학습을 시작합니다.",
                    "total_windows": total,
                    "counts": counts,
                    "epochs": epochs,
                }
                await _send_to_device_and_apps(
                    websocket,
                    user_id,
                    training_message,
                )

                x_calib_raw = np.stack(x_windows, axis=0)
                y_calib = np.array(y_labels, dtype=np.int64)

                print(
                    "🚀 [CALIB] x_calib_raw shape="
                    f"{x_calib_raw.shape}"
                )
                print(
                    "🚀 [CALIB] y_calib shape="
                    f"{y_calib.shape}"
                )
                print(
                    "🚀 [CALIB] unique labels="
                    f"{np.unique(y_calib, return_counts=True)}"
                )

                try:
                    result = await run_in_threadpool(
                        run_user_calibration,
                        user_id=user_id,
                        x_calib_raw=x_calib_raw,
                        y_calib=y_calib,
                        epochs=epochs,
                    )
                except Exception as error:
                    session["state"] = "failed"

                    response = {
                        "type": "training_failed",
                        "message": (
                            f"학습 중 오류가 발생했습니다: {error}"
                        ),
                    }
                    print(
                        f"🔥 [{user_id}] 학습 중 오류: {error}"
                    )
                    await _send_to_device_and_apps(
                        websocket,
                        user_id,
                        response,
                    )
                    continue

                session["state"] = "complete"

                response = {
                    "type": "complete",
                    "message": (
                        "캘리브레이션 및 사용자 모델 학습 완료"
                    ),
                    "result": result,
                }
                await _send_to_device_and_apps(
                    websocket,
                    user_id,
                    response,
                )

            elif message_type == "close":
                await websocket.send_json({
                    "type": "closed",
                    "message": "캘리브레이션 연결을 종료합니다.",
                })
                await websocket.close()
                break

            else:
                response = {
                    "type": "error",
                    "message": (
                        "알 수 없는 message type입니다: "
                        f"{message_type}"
                    ),
                }
                await _send_to_device_and_apps(
                    websocket,
                    user_id,
                    response,
                )

    except WebSocketDisconnect:
        print(
            f"❌ [{user_id}] 캘리브레이션 WebSocket "
            f"연결 끊김 role={role}"
        )

    except Exception as error:
        print(
            f"🔥 [{user_id}] 캘리브레이션 WebSocket 오류: "
            f"{error}"
        )
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(error),
            })
        except Exception:
            pass

    finally:
        if role == "device":
            if session.get("device_client") is websocket:
                session["device_client"] = None
        else:
            session["app_clients"].discard(websocket)

        if (
            session.get("device_client") is None
            and not session["app_clients"]
            and session["state"] in {
                "idle",
                "complete",
                "failed",
            }
        ):
            calibration_sessions.pop(user_id, None)
