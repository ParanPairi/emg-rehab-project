import time
import requests

from core.config import GEMINI_API_KEY


GEMINI_MODEL_NAME = "gemini-2.5-flash-lite"


def _extract_gemini_text(result: dict) -> str:
    candidates = result.get("candidates", [])

    if not candidates:
        return "AI 재활 조언 생성 실패: Gemini 응답 후보가 없습니다."

    candidate = candidates[0]
    finish_reason = candidate.get("finishReason")

    content = candidate.get("content", {})
    parts = content.get("parts", [])

    if not parts:
        return f"AI 재활 조언 생성 실패: Gemini 응답 내용이 없습니다. finishReason={finish_reason}"

    text = parts[0].get("text", "").strip()

    if not text:
        return f"AI 재활 조언 생성 실패: Gemini 응답 text가 없습니다. finishReason={finish_reason}"

    return text


def create_rehab_prescription(
    patient_name: str,
    fatigue_slope: float,
    comp_ratio: float,
    days_to_target: int,
) -> str:
    if not GEMINI_API_KEY:
        return "AI 조언 기능을 사용하려면 서버에 GEMINI_API_KEY가 설정되어야 합니다."

    prompt = f"""
당신은 친절하고 전문적인 AI 재활 작업치료사입니다.

다음은 현재 환자({patient_name})의 오늘 재활 운동 결과 데이터입니다.

[환자 생체 데이터]
- 근육 피로도 증가율: {fatigue_slope}
- 보상 작용 비율: {comp_ratio}
- 정상 근력까지 예상 회복일: {days_to_target}일

위 데이터를 바탕으로 환자에게 직접 말하듯 다정하고 격려하는 말투로 재활 조언을 작성해주세요.

조건:
1. 3~4문장으로 작성
2. 현재 상태에 대한 쉬운 요약 포함
3. 운동 강도, 휴식, 자세 교정 중 필요한 조언 포함
4. 앞으로의 회복 전망 포함
5. 전문적인 의학 용어는 피하고 쉽게 설명
6. 답변은 한국어로 작성
"""

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/{GEMINI_MODEL_NAME}:generateContent"
    )

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt}
                ],
            }
        ]
    }

    retry_status_codes = {429, 500, 502, 503, 504}
    last_error = ""

    for attempt in range(2):
        try:
            response = requests.post(
                url,
                params={"key": GEMINI_API_KEY},
                json=payload,
                timeout=25,
            )

            if response.status_code == 200:
                result = response.json()

                # 서버 터미널에서 응답 상태 확인용
                candidates = result.get("candidates", [])
                if candidates:
                    print("GEMINI finishReason:", candidates[0].get("finishReason"))
                print("GEMINI usageMetadata:", result.get("usageMetadata"))

                return _extract_gemini_text(result)

            last_error = response.text

            if response.status_code in retry_status_codes:
                print(f"Gemini temporary error {response.status_code}, retry {attempt + 1}/2")
                time.sleep(1)
                continue

            return f"AI 재활 조언 생성 실패: {response.status_code} / {response.text}"

        except requests.exceptions.Timeout:
            last_error = "요청 시간이 초과되었습니다."
            time.sleep(1)

        except requests.exceptions.RequestException as e:
            last_error = str(e)
            time.sleep(1)

        except Exception as e:
            return f"AI 재활 조언 생성 실패: {e}"

    print("GEMINI last_error:", last_error)
    return "현재 AI 조언 서버가 일시적으로 혼잡합니다. 잠시 후 다시 시도해주세요."