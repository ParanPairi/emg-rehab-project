from routers.emg_ws_router import router as emg_ws_router
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.predict_router import router as predict_router
from routers.logs_router import router as logs_router
from routers.recovery_router import router as recovery_router
from routers.prescription_router import router as prescription_router
from routers.tts_router import router as tts_router
from routers.calibration_router import router as calibration_router
from routers.device_router import router as device_router


app = FastAPI(title="EMGHAB CNN AI Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)
app.include_router(logs_router)
app.include_router(recovery_router)
app.include_router(prescription_router)
app.include_router(tts_router)
app.include_router(calibration_router)
app.include_router(emg_ws_router)
app.include_router(device_router)

print("DEVICE ROUTER LOADED")

@app.get("/")
def root():
    return {
        "message": "EMGHAB CNN AI + Firebase Server is Running!"
    }
