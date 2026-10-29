#!/bin/bash
# EMGHAB CNN AI Server 실행 스크립트

# tf_env 가상환경 활성화
source ./tf_env/bin/activate

# 서버 실행 (uvicorn 사용)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
