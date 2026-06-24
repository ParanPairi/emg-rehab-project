import serial
import asyncio
import websockets
import json
import time

SERIAL_PORT = "/dev/ttyUSB0"   # 안 되면 /dev/ttyACM0 로 바꾸기
BAUD_RATE = 115200

DEVICE_ID = "pi1"

WS_URL = "ws://52.78.30.13:8001/ws/emg"

async def send_emg():
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print("Arduino serial connected:", SERIAL_PORT)

    async with websockets.connect(WS_URL) as ws:
        print("WebSocket connected:", WS_URL)

        while True:
            line = ser.readline().decode(errors="ignore").strip()

            if not line or line.startswith("#") or line.startswith("time_ms"):
                continue

            parts = line.split(",")

            try:
                if len(parts) == 7:
                    # Arduino 형식: time_ms,label,ch0,ch1,ch2,ch3,ch4
                    ch0, ch1, ch2, ch3, ch4 = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6])
                elif len(parts) >= 5:
                    ch0, ch1, ch2, ch3, ch4 = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                elif len(parts) == 4:
                    ch0, ch1, ch2, ch3, ch4 = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), 1023
                else:
                    print("bad line:", line)
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
                print("sent:", data)

            except ValueError:
                print("parse error:", line)

asyncio.run(send_emg())
