from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse
import json
import time
import csv
import os

app = FastAPI()

CSV_FILE = "emg_data.csv"

latest_data = {
    "received_at": time.time(),
    "data": {
        "time": time.time(),
        "label": 0,
        "ch0": 0,
        "ch1": 0,
        "ch2": 0,
        "ch3": 0,
        "ch4": 1023
    }
}


def save_to_csv(data):
    file_exists = os.path.isfile(CSV_FILE)

    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["time", "label", "ch0", "ch1", "ch2", "ch3", "ch4"])

        writer.writerow([
            data.get("time"),
            data.get("label"),
            data.get("ch0"),
            data.get("ch1"),
            data.get("ch2"),
            data.get("ch3"),
            data.get("ch4")
        ])


@app.get("/")
def graph_page():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>AWS EMG Real-time Graph</title>
        <style>
            body { font-family: Arial; background: #111; color: white; text-align: center; }
            canvas { background: #222; border: 1px solid #555; margin-top: 20px; }
            .value { font-size: 20px; margin-top: 15px; }
        </style>
    </head>
    <body>
        <h1>AWS EMG Real-time Graph</h1>
        <div class="value" id="values">Waiting...</div>
        <canvas id="graph" width="1000" height="450"></canvas>

        <script>
            const canvas = document.getElementById("graph");
            const ctx = canvas.getContext("2d");
            const valuesDiv = document.getElementById("values");

            let data0 = [], data1 = [], data2 = [], data3 = [], data4 = [];
            const maxPoints = 200;

            async function fetchLatest() {
                const res = await fetch("/latest");
                const json = await res.json();

                if (!json.latest_data || !json.latest_data.data) return;

                const d = json.latest_data.data;

                data0.push(d.ch0);
                data1.push(d.ch1);
                data2.push(d.ch2);
                data3.push(d.ch3);
                data4.push(d.ch4);

                if (data0.length > maxPoints) {
                    data0.shift(); data1.shift(); data2.shift(); data3.shift(); data4.shift();
                }

                valuesDiv.innerHTML =
                    "label: " + d.label +
                    " | ch0: " + d.ch0 +
                    " | ch1: " + d.ch1 +
                    " | ch2: " + d.ch2 +
                    " | ch3: " + d.ch3 +
                    " | ch4: " + d.ch4;

                drawGraph();
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
                ctx.strokeStyle = "red"; drawLine(data0);
                ctx.strokeStyle = "lime"; drawLine(data1);
                ctx.strokeStyle = "cyan"; drawLine(data2);
                ctx.strokeStyle = "yellow"; drawLine(data3);
                ctx.strokeStyle = "white"; drawLine(data4);
            }

            setInterval(fetchLatest, 100);
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
            print("received:", msg)

            try:
                data = json.loads(msg)
            except Exception:
                data = {"raw": msg}

            latest_data = {
                "received_at": time.time(),
                "data": data
            }

            if all(k in data for k in ["time", "label", "ch0", "ch1", "ch2", "ch3", "ch4"]):
                save_to_csv(data)

            await websocket.send_text("received")

    except Exception as e:
        print("disconnected:", e)
