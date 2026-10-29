import os
from dotenv import load_dotenv


load_dotenv()
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

FIREBASE_KEY_PATH = "secrets/firebase-service-account.json"

CNN_MODEL_PATH = "/home/ubuntu/capstone_ai/model/cnn_gru_model_4ch_env.h5"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCALER_RAW_PATH = os.path.join(REPO_ROOT, "model", "scaler_raw_4ch_env.pkl")
SCALER_FEAT_PATH = os.path.join(REPO_ROOT, "model", "scaler_feat_4ch_env.pkl")
USER_MODELS_DIR = os.path.join(BASE_DIR, "models", "users")
WINDOW_SIZE = 25
CHANNEL_COUNT = 4

NORMALIZE_INPUT = True
NORMALIZE_DIVISOR = 1023.0

GESTURE_LABELS = [
    "Rest",
    "Fist",
    "Wrist Flexion",
    "Wrist Extension",
    "Elbow Flexion",
    "Elbow Extension",
]

TARGET_RMS = 600