from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import serial
import threading
import time
import json
import asyncio

app = FastAPI()

SERIAL_PORT = "/dev/ttyUSB1"
BAUD_RATE = 115200

# 라벨 번호: 0~5 바꿔가면서 사용
LABEL = 0

latest_data = {
    "time": time.time(),
    "label": LABEL,
    "ch0": 0,
    "ch1": 0,
    "ch2": 0,
    "ch3": 0,
    "ch4": 1023
}


def read_serial():
    global latest_data

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Arduino connected: {SERIAL_PORT}")

        while True:
            line = ser.readline().decode(errors="ignore").strip()

            if not line:
                continue

            # 아두이노 출력 예:
            # 512,498,530,490,1023
            parts = line.split(",")

            if len(parts) < 4:
                print("bad line:", line)
                continue

            try:
                ch0 = int(parts[0])
                ch1 = int(parts[1])
                ch2 = int(parts[2])
                ch3 = int(parts[3])

                # 5번째 채널이 들어오면 그 값 사용
                # 안 들어오면 1023 고정값 사용
                if len(parts) >= 5:
                    ch4 = int(parts[4])
                else:
                    ch4 = 1023

                latest_data = {
                    "time": time.time(),
                    "label": LABEL,
                    "ch0": ch0,
                    "ch1": ch1,
                    "ch2": ch2,
                    "ch3": ch3,
                    "ch4": ch4
                }

                print(latest_data)

            except ValueError:
                print("parse error:", line)

    except Exception as e:
        print("Serial error:", e)


@app.on_event("startup")
def startup_event():
    t = threading.Thread(target=read_serial, daemon=True)
    t.start()


@app.get("/")
def index():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>EMG 5CH Real-time Graph</title>
        <style>
            body {
                font-family: Arial;
                background: #111;
                color: white;
                text-align: center;
            }
            canvas {
                background: #222;
                border: 1px solid #555;
                margin-top: 20px;
            }
            .value {
                font-size: 20px;
                margin-top: 15px;
            }
        </style>
    </head>
    <body>
        <h1>EMG 5CH Real-time Graph</h1>
        <div class="value" id="values">Waiting...</div>
        <canvas id="graph" width="1000" height="450"></canvas>

        <script>
            const canvas = document.getElementById("graph");
            const ctx = canvas.getContext("2d");
            const valuesDiv = document.getElementById("values");

            let data0 = [];
            let data1 = [];
            let data2 = [];
            let data3 = [];
            let data4 = [];
            const maxPoints = 200;

            const ws = new WebSocket("ws://" + location.host + "/ws/emg");

            ws.onmessage = function(event) {
                const d = JSON.parse(event.data);

                data0.push(d.ch0);
                data1.push(d.ch1);
                data2.push(d.ch2);
                data3.push(d.ch3);
                data4.push(d.ch4);

                if (data0.length > maxPoints) {
                    data0.shift();
                    data1.shift();
                    data2.shift();
                    data3.shift();
                    data4.shift();
                }

                valuesDiv.innerHTML =
                    "label: " + d.label +
                    " | ch0: " + d.ch0 +
                    " | ch1: " + d.ch1 +
                    " | ch2: " + d.ch2 +
                    " | ch3: " + d.ch3 +
                    " | ch4: " + d.ch4;

                drawGraph();
            };

            function drawLine(data) {
                ctx.beginPath();
                for (let i = 0; i < data.length; i++) {
                    let x = i * (canvas.width / maxPoints);
                    let y = canvas.height - (data[i] / 1023) * canvas.height;

                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }
                ctx.stroke();
            }

            function drawGraph() {
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                ctx.strokeStyle = "red";
                drawLine(data0);

                ctx.strokeStyle = "lime";
                drawLine(data1);

                ctx.strokeStyle = "cyan";
                drawLine(data2);

                ctx.strokeStyle = "yellow";
                drawLine(data3);

                ctx.strokeStyle = "white";
                drawLine(data4);
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(html)


@app.get("/latest")
def get_latest():
    return latest_data


@app.websocket("/ws/emg")
async def websocket_emg(websocket: WebSocket):
    await websocket.accept()
    print("Web client connected")

    try:
        while True:
            await websocket.send_text(json.dumps(latest_data))
            await asyncio.sleep(0.05)

    except Exception as e:
        print("Web client disconnected:", e)
