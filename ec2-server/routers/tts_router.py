from io import BytesIO

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from schemas.emg_schema import TTSRequest
from services.tts_service import synthesize_korean_speech


router = APIRouter()


@router.post("/api/tts")
def create_tts_audio(data: TTSRequest):
    try:
        audio_bytes = synthesize_korean_speech(data.text)

        return StreamingResponse(
            BytesIO(audio_bytes),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "inline; filename=rehab_advice.mp3"
            },
        )

    except Exception as e:
        print("TTS 생성 실패:", e)
        raise HTTPException(
            status_code=500,
            detail={
                "message": "TTS 음성 생성 실패",
                "error": str(e),
            },
        )