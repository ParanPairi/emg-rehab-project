from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
import json
import time
from services.emg_stream_service import add_emg, get_stream_status

router = APIRouter()

latest_emg_data = None
active_user_id = None

# 대시보드(웹/앱) 클라이언트만 여기 담음. 센서(라즈베리파이)는 담지 않음.
dashboard_clients = set()


@router.post("/set-active-user/{user_id}")
def set_active_user(user_id: str):
    global active_user_id
    active_user_id = user_id
    return {"success": True, "activeUserId": active_user_id}


@router.get("/active-user")
def get_active_user():
    return {"activeUserId": active_user_id}


async def broadcast_to_dashboards(message: dict):
    """대시보드 클라이언트들에게만 전송 (센서엔 안 보냄)"""
    dead = []

    print(
        "[DASHBOARD BROADCAST]",
        f"clients={len(dashboard_clients)}",
        f"status={message.get('status')}",
    )

    for client in list(dashboard_clients):
        try:
            await client.send_json(message)
        except Exception:
            dead.append(client)
    for client in dead:
        dashboard_clients.discard(client)


# ─────────────────────────────────────
# 센서(라즈베리파이) 전용: 받기만 함
# ─────────────────────────────────────
@router.websocket("/ws/emg")
async def websocket_emg(websocket: WebSocket):
    global latest_emg_data

    await websocket.accept()
    print("EMG(센서) WebSocket connected")

    try:
        while True:
            data = await websocket.receive_text()

            print("[EMG SENSOR RAW]", data)

            try:
                parsed_data = json.loads(data)
            except Exception:
                parsed_data = {"raw": data}

            latest_emg_data = {
                "received_at": time.time(),
                "data": parsed_data,
            }

            # values 추출 (values > channels > ch0~ch4 순서)
            values = parsed_data.get("values")
            if values is None:
                values = parsed_data.get("channels")
            if values is None:
                ch_keys = ["ch0", "ch1", "ch2", "ch3"]
                if all(k in parsed_data for k in ch_keys):
                    values = [parsed_data[k] for k in ch_keys]

            # ─── 원본 EMG 값을 그래프용으로 대시보드에 전송 ───
            if values is not None and len(values) == 4:

                print(
                    "[EMG RAW READY]",
                    f"values={values}",
                    f"dashboard_clients={len(dashboard_clients)}",
                )

                await broadcast_to_dashboards({
                    "status": "emg_raw",
                    "ch0": values[0],
                    "ch1": values[1],
                    "ch2": values[2],
                    "ch3": values[3],
                    "time": time.time(),
                })
            else:
                print(
                    "[EMG RAW INVALID]",
                    f"values={values}",
                    f"length={len(values) if values is not None else None}",
                )

            result = None
            if values is not None:
                user_id = active_user_id or parsed_data.get("user_id") or "default"
                try:
                    # 무거운 작업(CNN/Firebase)을 스레드로 빼서 이벤트 루프 블로킹 방지
                    result = await asyncio.to_thread(add_emg, values, user_id)
                except Exception as pred_err:
                    print(f"[EMG] 예측 오류 (연결 유지): {pred_err}")

            # ─── 훈련 모드 카운트 판정 ───
            if result is not None:
                from services.training_service import process_prediction
                training_event = process_prediction(
                    gesture=result.get("gesture"),
                    confidence=result.get("confidence", 0.0),
                )
                if training_event is not None:
                    await broadcast_to_dashboards({
                        "status": "training",
                        "event": training_event,
                    })

            # ─── 예측 결과가 나왔을 때만 대시보드로 전송 ───
            if result is not None:
                await broadcast_to_dashboards({
                    "status": "prediction",
                    "activeUserId": active_user_id,
                    "prediction": result,
                })

    except WebSocketDisconnect:
        print("EMG(센서) WebSocket 정상 종료")
    except Exception as e:
        print("EMG(센서) WebSocket 오류:", e)
        try:
            await websocket.close()
        except Exception:
            pass


# ─────────────────────────────────────
# 대시보드(웹/앱) 전용: 결과/그래프 데이터 받음
# ─────────────────────────────────────
@router.websocket("/ws/dashboard")
async def websocket_dashboard(websocket: WebSocket):
    await websocket.accept()
    dashboard_clients.add(websocket)
    
    print(
        "[DASHBOARD CONNECTED]",
        f"client={websocket.client}",
        f"count={len(dashboard_clients)}",
    )

    try:
        while True:
            # 대시보드는 보통 받기만 하지만, 연결 유지를 위해 recv 대기
            await websocket.receive_text()
    except WebSocketDisconnect:
        print("Dashboard 정상 종료")
    except Exception as e:
        print("Dashboard 오류:", e)
    finally:
        dashboard_clients.discard(websocket)
        print("Dashboard 연결 해제. 현재:", len(dashboard_clients))


@router.get("/latest-emg")
def get_latest_emg():
    return {
        "activeUserId": active_user_id,
        "latest_emg_data": latest_emg_data,
        "connected_dashboards": len(dashboard_clients),
        **get_stream_status(),
    }