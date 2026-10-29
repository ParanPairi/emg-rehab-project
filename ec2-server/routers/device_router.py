from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
import asyncio
import time

router = APIRouter()

device_clients = {}
device_status = {}


@router.websocket("/ws/device/{device_id}")
async def device_websocket(websocket: WebSocket, device_id: str):
    await websocket.accept()

    device_clients[device_id] = websocket
    device_status[device_id] = {
        "connected": True,
        "measuring": False,
        "last_seen": time.time(),
    }

    print(f"라즈베리 연결됨: {device_id}")

    try:
        while True:
            # 30초까지 메시지가 없어도 연결을 끊지 않고 계속 대기
            try:
                data = await asyncio.wait_for(
                    websocket.receive_json(), timeout=30.0
                )
            except asyncio.TimeoutError:
                continue

            print(f"[{device_id}] 상태 수신:", data)
            device_status[device_id]["last_seen"] = time.time()

            if data.get("type") == "status":
                device_status[device_id]["measuring"] = bool(
                    data.get("measuring", False)
                )
                if "mode" in data:
                    device_status[device_id]["mode"] = data.get("mode")
                # 응답을 돌려줘 양방향 통신 유지 (연결 안정화)
                try:
                    await websocket.send_json({"type": "ack"})
                except Exception:
                    pass

    except WebSocketDisconnect:
        print(f"라즈베리 연결 끊김: {device_id}")

    except Exception as e:
        print(f"라즈베리 오류 [{device_id}]:", e)

    finally:
        device_clients.pop(device_id, None)
        device_status[device_id] = {
            "connected": False,
            "measuring": False,
            "last_seen": time.time(),
        }


@router.post("/device/start/{device_id}")
async def start_device(
    device_id: str,
    mode: str = "training",
    user_id: str | None = None,
):
    ws = device_clients.get(device_id)

    if ws is None:
        raise HTTPException(
            status_code=404,
            detail="라즈베리가 서버에 연결되어 있지 않습니다.",
        )

    try:
        await ws.send_json({
            "type": "start",
            "mode": mode,               # training 또는 calibration
            "device_id": device_id,
            "user_id": user_id,
        })
    except Exception:
        device_clients.pop(device_id, None)
        device_status[device_id] = {
            "connected": False,
            "measuring": False,
            "last_seen": time.time(),
        }
        raise HTTPException(
            status_code=404,
            detail="라즈베리 연결이 끊겨 있습니다. 라즈베리 코드를 다시 실행하세요.",
        )

    device_status.setdefault(device_id, {})
    device_status[device_id]["connected"] = True
    device_status[device_id]["measuring"] = True
    device_status[device_id]["mode"] = mode
    device_status[device_id]["last_seen"] = time.time()

    return {
        "success": True,
        "message": f"{mode} 시작 명령 전송 완료",
        "device_id": device_id,
        "mode": mode,
    }


@router.post("/device/stop/{device_id}")
async def stop_device(device_id: str):
    ws = device_clients.get(device_id)

    if ws is None:
        raise HTTPException(
            status_code=404,
            detail="라즈베리가 서버에 연결되어 있지 않습니다.",
        )

    try:
        await ws.send_json({
            "type": "stop",
            "device_id": device_id,
        })
    except Exception:
        device_clients.pop(device_id, None)
        device_status[device_id] = {
            "connected": False,
            "measuring": False,
            "last_seen": time.time(),
        }
        raise HTTPException(
            status_code=404,
            detail="라즈베리 연결이 끊겨 있습니다. 라즈베리 코드를 다시 실행하세요.",
        )

    device_status.setdefault(device_id, {})
    device_status[device_id]["connected"] = True
    device_status[device_id]["measuring"] = False
    device_status[device_id]["last_seen"] = time.time()

    return {
        "success": True,
        "message": "측정 정지 명령 전송 완료",
        "device_id": device_id,
    }


@router.get("/device/status")
def get_device_status():
    return {
        "devices": device_status,
        "connected_devices": list(device_clients.keys()),
    }