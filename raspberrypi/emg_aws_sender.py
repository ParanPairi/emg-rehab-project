import asyncio
import time
import json
import serial
import websockets
from serial import SerialException

SERIAL_PORT = "/dev/ttyUSB0"   # 안 되면 /dev/ttyACM0
BAUD_RATE = 115200

DEVICE_ID = "pi1"

AWS_EMG_WS_URL = "ws://52.78.30.13:8001/ws/emg"
AWS_DEVICE_WS_URL = f"ws://52.78.30.13:8001/ws/device/{DEVICE_ID}"

SEND_INTERVAL_SEC = 0.05

measuring = False


async def device_control_loop():
    global measuring

    while True:
        try:
            print("Connecting device control:", AWS_DEVICE_WS_URL)

            async with websockets.connect(
                AWS_DEVICE_WS_URL,
                ping_interval=20,
                ping_timeout=20
            ) as ws:
                print("Device control connected")

                await ws.send(json.dumps({
                    "type": "status",
                    "device_id": DEVICE_ID,
                    "measuring": measuring,
                    "time": time.time()
                }))

                while True:
                    msg = await ws.recv()
                    print("device command:", msg)

                    try:
                        data = json.loads(msg)
                    except json.JSONDecodeError:
                        print("invalid command:", msg)
                        continue

                    cmd = data.get("type")

                    if cmd == "start":
                        measuring = True
                        print("측정 시작")

                    elif cmd == "stop":
                        measuring = False
                        print("측정 정지")

                    await ws.send(json.dumps({
                        "type": "status",
                        "device_id": DEVICE_ID,
                        "measuring": measuring,
                        "time": time.time()
                    }))

        except Exception as e:
            print("Device control reconnect:", e)
            await asyncio.sleep(2)


def open_serial():
    while True:
        try:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            time.sleep(2)
            ser.reset_input_buffer()
            print("Arduino connected:", SERIAL_PORT)
            return ser
        except Exception as e:
            print("Arduino connect failed:", e)
            time.sleep(2)


async def emg_sender_loop():
    global measuring

    while True:
        ser = open_serial()

        try:
            print("Connecting EMG WebSocket:", AWS_EMG_WS_URL)

            async with websockets.connect(
                AWS_EMG_WS_URL,
                ping_interval=20,
                ping_timeout=20
            ) as emg_ws:
                print("EMG WebSocket connected")

                last_sent_time = 0.0

                while True:
                    try:
                        raw = ser.readline()
                    except SerialException as e:
                        print("Serial disconnected:", e)
                        break

                    line = raw.decode(errors="ignore").strip()

                    if not line:
                        await asyncio.sleep(0)
                        continue

                    if line.startswith("#"):
                        print(line)
                        continue

                    if line.startswith("time_ms"):
                        print(line)
                        continue

                    parts = line.split(",")

                    if len(parts) != 6:
                        print("skip invalid format:", line)
                        continue

                    try:
                        time_ms = int(parts[0])
                        ch0 = int(parts[1])
                        ch1 = int(parts[2])
                        ch2 = int(parts[3])
                        ch3 = int(parts[4])
                        ch4 = int(parts[5])
                    except ValueError:
                        print("skip invalid value:", line)
                        continue

                    if not measuring:
                        continue

                    now = time.time()

                    if now - last_sent_time < SEND_INTERVAL_SEC:
                        continue

                    last_sent_time = now

                    payload = {
                        "type": "emg",
                        "time": now,
                        "device_id": DEVICE_ID,
                        "time_ms": time_ms,
                        "values": [ch0, ch1, ch2, ch3, ch4],
                        "ch0": ch0,
                        "ch1": ch1,
                        "ch2": ch2,
                        "ch3": ch3,
                        "ch4": ch4,
                    }

                    await emg_ws.send(json.dumps(payload))
                    print("sent:", payload)

                    await asyncio.sleep(0)

        except Exception as e:
            print("EMG WebSocket reconnect:", e)

        try:
            ser.close()
        except:
            pass

        await asyncio.sleep(2)


async def main():
    await asyncio.gather(
        device_control_loop(),
        emg_sender_loop()
    )


if __name__ == "__main__":
    asyncio.run(main())
