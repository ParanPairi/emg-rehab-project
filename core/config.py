import os
from dotenv import load_dotenv


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

FIREBASE_KEY_PATH = "secrets/firebase-service-account.json"

CNN_MODEL_PATH = "/home/ubuntu/capstone_ai/model/cnn_gru_model_5ch_env.h5"

WINDOW_SIZE = 25
CHANNEL_COUNT = 5

NORMALIZE_INPUT = True
NORMALIZE_DIVISOR = 1023.0

GESTURE_LABELS = [
    "Rest",
    "Fist",
    "Wrist Flexion",
    "Wrist Extension",
    "Radial Deviation",
    "Ulnar Deviation",
]

TARGET_RMS = 600