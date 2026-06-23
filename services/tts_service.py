import base64
import re
import requests

from google.oauth2 import service_account
from google.auth.transport.requests import Request

from core.config import FIREBASE_KEY_PATH


TTS_SCOPE = ["https://www.googleapis.com/auth/cloud-platform"]
TTS_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"

TTS_VOICE_NAME = "ko-KR-Neural2-A"


def _get_access_token() -> str:
    credentials = service_account.Credentials.from_service_account_file(
        FIREBASE_KEY_PATH,
        scopes=TTS_SCOPE,
    )
    credentials.refresh(Request())
    return credentials.token


def _clean_text_for_tts(text: str) -> str:
    cleaned = text.strip()

    emoji_pattern = re.compile(
        "["
        "\U0001F300-\U0001F5FF"
        "\U0001F600-\U0001F64F"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U00002700-\U000027BF"
        "\U00002600-\U000026FF"
        "]+",
        flags=re.UNICODE,
    )
    cleaned = emoji_pattern.sub("", cleaned)

    # 마크다운/장식 기호 제거
    cleaned = cleaned.replace("💪", "")
    cleaned = cleaned.replace("🔥", "")
    cleaned = cleaned.replace("✅", "")
    cleaned = cleaned.replace("⭐", "")
    cleaned = cleaned.replace("*", "")
    cleaned = cleaned.replace("#", "")
    cleaned = cleaned.replace("•", "")
    cleaned = cleaned.replace("·", ", ")

    # URL 제거
    cleaned = re.sub(r"https?://\S+", "", cleaned)

    # TTS가 이상하게 읽는 기호 정리
    cleaned = cleaned.replace("/", " 또는 ")
    cleaned = cleaned.replace("&", " 그리고 ")
    cleaned = cleaned.replace("_", " ")
    cleaned = cleaned.replace("-", " ")

    # 괄호/따옴표 일부 정리
    cleaned = cleaned.replace("(", " ")
    cleaned = cleaned.replace(")", " ")
    cleaned = cleaned.replace("[", " ")
    cleaned = cleaned.replace("]", " ")
    cleaned = cleaned.replace("{", " ")
    cleaned = cleaned.replace("}", " ")

    # 여러 공백 정리
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # 너무 긴 텍스트 방지
    if len(cleaned) > 1200:
        cleaned = cleaned[:1200]

    return cleaned


def _request_tts(access_token: str, text: str) -> requests.Response:
    payload = {
        "input": {
            "text": text,
        },
        "voice": {
            "languageCode": "ko-KR",
            "name": TTS_VOICE_NAME,
        },
        "audioConfig": {
            "audioEncoding": "MP3",
            "speakingRate": 0.92,
            "pitch": -1.5,
        },
    }

    return requests.post(
        TTS_URL,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )


def synthesize_korean_speech(text: str) -> bytes:
    cleaned_text = _clean_text_for_tts(text)

    if not cleaned_text:
        raise ValueError("TTS로 변환할 텍스트가 비어 있습니다.")

    access_token = _get_access_token()

    response = _request_tts(
        access_token=access_token,
        text=cleaned_text,
    )

    if response.status_code != 200:
        print(f"TTS failed: {response.status_code} / {response.text}")
        raise RuntimeError(
            f"Google TTS 생성 실패: {response.status_code} / {response.text}"
        )

    result = response.json()
    audio_content = result.get("audioContent")

    if not audio_content:
        raise RuntimeError("Google TTS 응답에 audioContent가 없습니다.")

    print(f"TTS voice used: {TTS_VOICE_NAME}")

    return base64.b64decode(audio_content)