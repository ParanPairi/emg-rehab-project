import time

# 훈련 세션 상태 (단일 사용자 기준. 다중이면 dict로 확장)
training_session = {
    "active": False,
    "user_id": None,
    "target_gesture": None,
    "target_count": 0,
    "current_count": 0,
    "state": "idle",       # idle | waiting | in_motion
    "started_at": None,
    "last_count_at": None,
}

# 카운트 판정용 설정
CONFIDENCE_THRESHOLD = 60.0   # 이 신뢰도 이상이어야 동작으로 인정 (%)


def start_training(user_id: str, target_gesture: str, target_count: int) -> dict:
    """훈련 세션 시작"""
    global training_session
    training_session = {
        "active": True,
        "user_id": user_id,
        "target_gesture": target_gesture,
        "target_count": target_count,
        "current_count": 0,
        "state": "waiting",     # 목표 동작을 기다리는 상태
        "started_at": time.time(),
        "last_count_at": None,
    }
    print(f"🏋️ 훈련 시작: {user_id} / {target_gesture} / 목표 {target_count}회", flush=True)
    return get_training_status()


def stop_training() -> dict:
    """훈련 세션 종료"""
    global training_session
    training_session["active"] = False
    training_session["state"] = "idle"
    print("🛑 훈련 종료", flush=True)
    return get_training_status()


def process_prediction(gesture: str, confidence: float) -> dict | None:
    """
    예측 결과 1건을 받아 카운트 판정.
    카운트가 늘거나 상태가 바뀌면 이벤트 dict 반환, 아니면 None.
    상태머신: waiting → (목표동작) → in_motion → (Rest) → 카운트+1 → waiting
    """
    global training_session

    if not training_session["active"]:
        return None

    # ─── 디버그 로그 ───────────────────────────────
    print(
        f"[훈련] 예측={gesture}, 신뢰도={confidence:.1f}%, "
        f"상태={training_session['state']}, "
        f"목표={training_session['target_gesture']}, "
        f"카운트={training_session['current_count']}/{training_session['target_count']}",
        flush=True,
    )
    # ──────────────────────────────────────────────

    target = training_session["target_gesture"]
    state = training_session["state"]

    # 신뢰도 낮으면 무시
    if confidence < CONFIDENCE_THRESHOLD:
        print(f"[훈련] → 신뢰도 미달({confidence:.1f}% < {CONFIDENCE_THRESHOLD}%), 무시", flush=True)
        return None

    event = None

    if state == "waiting":
        # 목표 동작을 인식하면 → 동작 중 상태로
        if gesture == target:
            training_session["state"] = "in_motion"
            event = {"type": "motion_started", "gesture": gesture}
            print(f"[훈련] → 목표동작 감지! in_motion 진입", flush=True)

    elif state == "in_motion":
        # Rest로 복귀하면 → 1회 카운트
        if gesture == "Rest":
            training_session["current_count"] += 1
            training_session["last_count_at"] = time.time()
            training_session["state"] = "waiting"

            count = training_session["current_count"]
            target_count = training_session["target_count"]

            print(f"[훈련] → Rest 복귀! 카운트 +1 = {count}/{target_count}", flush=True)

            event = {
                "type": "count",
                "current_count": count,
                "target_count": target_count,
            }

            # 목표 도달 → 훈련 완료
            if count >= target_count:
                training_session["active"] = False
                training_session["state"] = "idle"
                print(f"[훈련] → 🎉 목표 달성! 훈련 완료", flush=True)
                event = {
                    "type": "completed",
                    "current_count": count,
                    "target_count": target_count,
                }

    return event


def get_training_status() -> dict:
    return {
        "active": training_session["active"],
        "user_id": training_session["user_id"],
        "target_gesture": training_session["target_gesture"],
        "target_count": training_session["target_count"],
        "current_count": training_session["current_count"],
        "state": training_session["state"],
    }