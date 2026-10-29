from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse
import json
import time
import csv
import os

app = FastAPI()

# 이전 5채널 데이터와 섞이지 않도록 파일명 변경 권장
CSV_FILE = "emg_data_4ch.csv" 

latest_data = {
    "received_at": 0,
    "data": {
        "label": 0,
        "ch0": 0,
        "ch1": 0,
        "ch2": 0,
        "ch3": 0
    }
}

def save_to_csv(data):
    file_exists = os.path.isfile(CSV_FILE)

    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)

        # 🚨 수정 포인트 1: AI 학습 코드와 100% 동일한 헤더 순서로 맞춤
        if not file_exists:
            writer.writerow(["ch0", "ch1", "ch2", "ch3", "label"])

        writer.writerow([
            data.get("ch0", 0),
            data.get("ch1", 0),
            data.get("ch2", 0),
            data.get("ch3", 0),
            data.get("label", 0)
        ])

@app.get("/")
def graph_page():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>AWS EMG Real-time Graph (4CH)</title>
        <style>
            body { font-family: Arial; background: #111; color: white; text-align: center; }
            canvas { background: #222; border: 1px solid #555; margin-top: 20px; }
            .value { font-size: 20px; margin-top: 15px; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>AWS EMG Real-time Graph (4 Channels)</h1>
        <div class="value" id="values">Waiting for sensor data...</div>
        <canvas id="graph" width="1000" height="450"></canvas>

        <script>
            const canvas = document.getElementById("graph");
            const ctx = canvas.getContext("2d");
            const valuesDiv = document.getElementById("values");

            let data0 = [], data1 = [], data2 = [], data3 = [];
            const maxPoints = 200;
            let lastReceivedAt = 0; // 🚨 수정 포인트 2: 중복 데이터 렌더링 방지용

            async function fetchLatest() {
                try {
                    const res = await fetch("/latest");
                    const json = await res.json();

                    if (!json.latest_data || !json.latest_data.data) return;

                    // 서버에서 새 데이터가 오지 않았으면 그래프를 멈춰서 자연스럽게 표현
                    if (json.latest_data.received_at === lastReceivedAt) return;
                    lastReceivedAt = json.latest_data.received_at;

                    const d = json.latest_data.data;

                    data0.push(d.ch0);
                    data1.push(d.ch1);
                    data2.push(d.ch2);
                    data3.push(d.ch3);

                    if (data0.length > maxPoints) {
                        data0.shift(); data1.shift(); data2.shift(); data3.shift();
                    }

                    valuesDiv.innerHTML =
                        "Label: " + d.label +
                        " | ch0: " + d.ch0 +
                        " | ch1: " + d.ch1 +
                        " | ch2: " + d.ch2 +
                        " | ch3: " + d.ch3;

                    drawGraph();
                } catch (e) {
                    console.error("데이터 로드 에러:", e);
                }
            }

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
                ctx.lineWidth = 2;
                ctx.strokeStyle = "#ff4d4d"; drawLine(data0);
                ctx.strokeStyle = "#00ff00"; drawLine(data1);
                ctx.strokeStyle = "#00ccff"; drawLine(data2);
                ctx.strokeStyle = "#ffcc00"; drawLine(data3);
            }

            // 부드러운 그래프 출력을 위해 폴링 주기를 50ms로 단축
            setInterval(fetchLatest, 50);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(html)

@app.get("/latest")
def latest():
    return JSONResponse(content={"latest_data": latest_data})

@app.websocket("/ws/emg")
async def ws_emg(websocket: WebSocket):
    global latest_data

    await websocket.accept()
    print("Raspberry Pi connected")

    try:
        while True:
            msg = await websocket.receive_text()
            
            try:
                data = json.loads(msg)
            except Exception:
                data = {"raw": msg}

            latest_data = {
                "received_at": time.time(),
                "data": data
            }

            # 🚨 수정 포인트 3: "time" 검사를 빼서, RPi가 time을 안 보내도 CSV가 정상 저장되도록 고침
            required_keys = ["label", "ch0", "ch1", "ch2", "ch3"]
            if all(k in data for k in required_keys):
                save_to_csv(data)

            await websocket.send_text("received")

    except Exception as e:
        print("disconnected:", e)