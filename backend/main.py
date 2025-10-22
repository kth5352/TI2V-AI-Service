# backend/main.py
"""
FastAPI 엔드포인트 레이어
- 생성 요청(POST /api/generate)
- 상태 조회(GET /api/status/{job_id})
- SSE 스트림(GET /api/stream/{job_id})
- 결과 파일 서빙(GET /api/video/{filename})
- 헬스체크(GET /healthz)

리팩토링 포인트
1) 명시적 Pydantic 모델과 타입힌트로 응답 스키마 고정
2) 에러 응답에서도 error_code 표준화(비즈니스 레이어(video_worker)와 일치)
3) 공통 헤더/설정 상수화, CORS 도메인은 환경변수로 override 가능
4) 반복문 스트림에서 변화가 있을 때만 push + done/error 시 정상 종료
5) 상태코드/메시지는 FastAPI 표준 예외로 통일
"""

from __future__ import annotations

import os
import json
import asyncio
from typing import Optional

from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, PositiveInt, PositiveFloat

from video_worker import start_job, JOBS, MEDIA_DIR

# ---------------------------------------------------------------------
# 설정 상수
# ---------------------------------------------------------------------
ALLOWED_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
APP_TITLE = os.getenv("APP_TITLE", "WAN2.2 I2V Service")

# ---------------------------------------------------------------------
# FastAPI 앱
# ---------------------------------------------------------------------
app = FastAPI(title=APP_TITLE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # 배포 시 도메인 한정 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# Pydantic 모델 (응답/요청 스키마)
# ---------------------------------------------------------------------
class GenerateResp(BaseModel):
    """작업 생성 응답"""
    job_id: str = Field(..., description="추후 상태 조회/스트림 구독에 사용하는 ID")

class StatusResp(BaseModel):
    """상태 조회 응답"""
    status: str = Field(..., description="queued|running|done|error")
    progress: float = Field(..., ge=0.0, le=1.0)
    result: Optional[str] = Field(None, description="완료 시 /api/video/{filename}")
    error: Optional[str] = Field(None, description="에러 메시지(내부 로그와 동일하지 않을 수 있음)")
    error_code: Optional[str] = Field(None, description="표준화된 에러 코드")

# ---------------------------------------------------------------------
# 라우트
# ---------------------------------------------------------------------
@app.post("/api/generate", response_model=GenerateResp)
async def generate(
    prompt: str = Form(..., description="텍스트 프롬프트"),
    duration: float = Form(4.0, description="초 단위 길이(예: 4.0)"),
    fps: int = Form(16, description="초당 프레임"),
    width: int = Form(1280, description="원본 요청 너비"),
    height: int = Form(704, description="원본 요청 높이"),
    negative_prompt: Optional[str] = Form(None, description="네거티브 프롬프트"),
    image: Optional[UploadFile] = None,
):
    """
    비디오 생성 잡을 시작한다.
    - image가 있으면 I2V, 없으면 T2V
    - 상세 파이프라인/진행률/에러코드는 video_worker에서 관리
    """
    img_bytes = await image.read() if image is not None else None

    # 간단한 유효성 검증(너무 공격적으로 제한하지 않음)
    if not prompt or not prompt.strip():
        raise HTTPException(status_code=422, detail="prompt is required")

    if fps <= 0:
        raise HTTPException(status_code=422, detail="fps must be positive")

    if width <= 0 or height <= 0:
        raise HTTPException(status_code=422, detail="width/height must be positive")

    job_id = start_job(
        prompt=prompt.strip(),
        image_bytes=img_bytes,
        width=width,
        height=height,
        fps=fps,
        duration_sec=float(duration),
        negative_prompt=negative_prompt.strip() if negative_prompt else None,
    )
    return GenerateResp(job_id=job_id)


@app.get("/api/status/{job_id}", response_model=StatusResp)
def status(job_id: str):
    """
    현재 잡 상태를 즉시 스냅샷으로 돌려준다.
    - 프론트는 폴링 또는 /api/stream/{job_id} SSE를 활용한다.
    """
    info = JOBS.get(job_id)
    if not info:
        raise HTTPException(status_code=404, detail="job not found")

    resp: dict = {
        "status": info["status"],
        "progress": float(info.get("progress", 0.0)),
        "result": None,
        "error": info.get("error"),
        "error_code": info.get("error_code"),
    }
    if info["status"] == "done" and info.get("result_path"):
        fname = os.path.basename(info["result_path"])
        resp["result"] = f"/api/video/{fname}"

    return JSONResponse(resp)


@app.get("/api/stream/{job_id}")
async def stream(job_id: str):
    """
    Server-Sent Events
    - 이벤트 타입: "update"
    - 데이터: StatusResp 형태(JSON)
    - 상태 변화가 있을 때만 푸시
    - done/error 시 스트림 종료
    """
    async def event_gen():
        last_payload = None
        while True:
            info = JOBS.get(job_id)
            if not info:
                # 존재하지 않으면 스트림 종료
                yield f"event: update\ndata: {json.dumps({'status':'gone','progress':0.0})}\n\n"
                return

            payload = {
                "status": info["status"],
                "progress": float(info.get("progress", 0.0)),
                "result": None,
                "error": info.get("error"),
                "error_code": info.get("error_code"),
            }
            if info["status"] == "done" and info.get("result_path"):
                fname = os.path.basename(info["result_path"])
                payload["result"] = f"/api/video/{fname}"

            if payload != last_payload:
                yield f"event: update\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                last_payload = payload

            if info["status"] in ("done", "error"):
                return

            await asyncio.sleep(1.0)  # 푸시 주기

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",  # nginx에서 버퍼링 방지 옵션 사용 시
    }
    return StreamingResponse(event_gen(), media_type="text/event-stream", headers=headers)


@app.get("/api/video/{filename}")
def get_video(filename: str):
    """
    생성된 mp4를 서빙한다.
    - 실제 파일시스템 경로는 video_worker.MEDIA_DIR 하위
    """
    path = os.path.join(MEDIA_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(path, media_type="video/mp4", filename=filename)


@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    """간단한 헬스 엔드포인트 (L4/L7 헬스체크용)"""
    return "OK"
