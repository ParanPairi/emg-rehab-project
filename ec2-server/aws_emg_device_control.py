import asyncio
import websockets
import serial
import json
import time

SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200

DEVICE_ID = "pi1"

AWS_EMG_WS_URL = "ws://52.78.30.13:8000/ws/emg"
AWS_DEVICE_WS_URL = f"ws://52.78.30.13:8000/ws/device/{DEVICE_ID}"

measuring = False
prev_measuring = False


async def device_control_loop():
    global measuring

    while True:
        try:
            print("Connecting device control:", AWS_DEVICE_WS_URL)

            async with websockets.connect(
                AWS_DEVICE_WS_URL,
                ping_interval=None
            ) as ws:
                print("Device control connected")

                await ws.send(json.dumps({
                    "type": "status",
                    "device_id": DEVICE_ID,
                    "measuring": measuring,
                    "time": time.time(),
                }))

                async for message in ws:
                    data = json.loads(message)
                    command = data.get("type")

                    print("command received:", data)

                    if command == "start":
                        measuring = True
                        print("[START] measuring started")

                        await ws.send(json.dumps({
                            "type": "status",
                            "device_id": DEVICE_ID,
                            "measuring": True,
                            "time": time.time(),
                        }))

                    elif command == "stop":
                        measuring = False
                        print("[STOP] measuring stopped")

                        await ws.send(json.dumps({
                            "type": "status",
                            "device_id": DEVICE_ID,
                            "measuring": False,
                            "time": time.time(),
                        }))

        except Exception as e:
            print("Device control disconnected:", e)
            print("Reconnect device control after 3 seconds...")
            await asyncio.sleep(3)


async def emg_send_loop():
    global measuring, prev_measuring

    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print("Arduino connected:", SERIAL_PORT)

    while True:
        try:
            print("Connecting EMG WebSocket:", AWS_EMG_WS_URL)

            async with websockets.connect(
                AWS_EMG_WS_URL,
                ping_interval=None
            ) as ws:
                print("EMG WebSocket connected")

                async def drain_server_responses():
                    try:
                        async for _ in ws:
                            pass
                    except Exception:
                        pass

                asyncio.ensure_future(drain_server_responses())

                while True:
                    if not measuring:
                        prev_measuring = False
                        await asyncio.sleep(0.05)
                        continue

                    if not prev_measuring and measuring:
                        ser.reset_input_buffer()
                        print("[BUFFER] serial buffer cleared")
                        prev_measuring = True

                    line = ser.readline().decode(errors="ignore").strip()
                    await asyncio.sleep(0)

                    if not line:
                        continue

                    # 주석/헤더 라인 무시
                    if line.startswith("#") or line.startswith("time_ms"):
                        continue

                    try:
                        parts = line.split(",")

                        if len(parts) == 7:
                            # Arduino 형식: time_ms,label,ch0,ch1,ch2,ch3,ch4
                            ch0 = int(parts[2])
                            ch1 = int(parts[3])
                            ch2 = int(parts[4])
                            ch3 = int(parts[5])
                            ch4 = int(parts[6])

                        elif len(parts) >= 5:
                            # 단순 형식: ch0,ch1,ch2,ch3,ch4
                            ch0 = int(parts[0])
                            ch1 = int(parts[1])
                            ch2 = int(parts[2])
                            ch3 = int(parts[3])
                            ch4 = int(parts[4])

                        elif len(parts) == 4:
                            ch0 = int(parts[0])
                            ch1 = int(parts[1])
                            ch2 = int(parts[2])
                            ch3 = int(parts[3])
                            ch4 = 1023

                        else:
                            continue

                        data = {
                            "time": time.time(),
                            "device_id": DEVICE_ID,
                            "values": [ch0, ch1, ch2, ch3, ch4],
                            "ch0": ch0,
                            "ch1": ch1,
                            "ch2": ch2,
                            "ch3": ch3,
                            "ch4": ch4,
                        }

                        await ws.send(json.dumps(data))
                        print(f"[SENT] ch0:{ch0:4d} | ch1:{ch1:4d} | ch2:{ch2:4d} | ch3:{ch3:4d} | ch4:{ch4:4d}")

                    except ValueError:
                        # 숫자로 변환 안 되는 줄은 조용히 무시
                        continue

        except Exception as e:
            print("EMG WebSocket disconnected:", e)
            print("Reconnect EMG WebSocket after 3 seconds...")
            await asyncio.sleep(3)


async def main():
    await asyncio.gather(
        device_control_loop(),
        emg_send_loop(),
    )


if __name__ == "__main__":
    asyncio.run(main())
