from fastapi import APIRouter, WebSocket
import json
import time
from services.emg_stream_service import add_emg, get_stream_status

router = APIRouter()

latest_emg_data = None
active_user_id = None

connected_clients = set()


@router.post("/set-active-user/{user_id}")
def set_active_user(user_id: str):
    global active_user_id
    active_user_id = user_id

    return {
        "success": True,
        "activeUserId": active_user_id
    }


@router.get("/active-user")
def get_active_user():
    return {
        "activeUserId": active_user_id
    }


@router.websocket("/ws/emg")
async def websocket_emg(websocket: WebSocket):
    global latest_emg_data

    await websocket.accept()
    connected_clients.add(websocket)

    print("EMG WebSocket connected")
    print("현재 연결된 클라이언트 수:", len(connected_clients))

    try:
        while True:
            data = await websocket.receive_text()
            print("받은 EMG 데이터:", data)

            try:
                parsed_data = json.loads(data)
            except Exception:
                parsed_data = {"raw": data}

            latest_emg_data = {
                "received_at": time.time(),
                "data": parsed_data
            }

            values = parsed_data.get("values")

            if values is None:
                values = parsed_data.get("channels")

            if values is None:
                ch_keys = ["ch0", "ch1", "ch2", "ch3", "ch4"]
                if all(k in parsed_data for k in ch_keys):
                    values = [parsed_data[k] for k in ch_keys]

            result = None

            if values is not None:
                user_id = active_user_id or parsed_data.get("user_id") or "default"
                try:
                    result = add_emg(values, user_id)
                except Exception as pred_err:
                    print(f"[EMG] 예측 오류 (연결 유지): {pred_err}")

            message = {
                "status": "received",
                "activeUserId": active_user_id,
                "values": values,
                "prediction": result
            }

            disconnected_clients = []

            for client in connected_clients:
                try:
                    await client.send_json(message)
                except Exception:
                    disconnected_clients.append(client)

            for client in disconnected_clients:
                connected_clients.discard(client)

    except Exception as e:
        print("EMG WebSocket disconnected or error:", e)
        try:
            await websocket.close()
        except Exception:
            pass

    finally:
        connected_clients.discard(websocket)
        print("현재 연결된 클라이언트 수:", len(connected_clients))


@router.get("/latest-emg")
def get_latest_emg():
    return {
        "activeUserId": active_user_id,
        "latest_emg_data": latest_emg_data,
        "connected_clients": len(connected_clients),
        **get_stream_status(),
    }