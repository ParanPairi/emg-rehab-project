import serial
import asyncio
import websockets
import json
import time

# 아두이노 포트
SERIAL_PORT = "/dev/ttyUSB1"   # 안 되면 /dev/ttyACM0 로 바꾸기
BAUD_RATE = 115200

# 웹소켓 서버 주소
WS_URL = "ws://52.78.30.13:8002/ws/emg"

# 현재 동작 라벨
# 예: 0=주먹, 1=손펴기, 2=손목굽힘 ...
LABEL = 0

async def send_emg():
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print("Arduino serial connected:", SERIAL_PORT)

    async with websockets.connect(WS_URL) as ws:
        print("WebSocket connected:", WS_URL)

        while True:
            line = ser.readline().decode(errors="ignore").strip()

            if not line:
                continue

            # 아두이노에서 예: 512,498,501,520,1023 이렇게 온다고 가정
            parts = line.split(",")

            # 최소 4채널 이상 있어야 함
            if len(parts) < 4:
                print("bad line:", line)
                continue

            try:
                ch0 = int(parts[0])
                ch1 = int(parts[1])
                ch2 = int(parts[2])
                ch3 = int(parts[3])

                # 1023 채널은 아예 안 보냄
                data = {
                    "time": time.time(),
                    "label": LABEL,
                    "channels": [ch0, ch1, ch2, ch3]
                }

                await ws.send(json.dumps(data))
                print("sent:", data)

            except ValueError:
                print("parse error:", line)

asyncio.run(send_emg())
