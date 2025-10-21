# backend/main.py — status와 stream 응답에 error_code 포함
import os
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import json

from video_worker import start_job, JOBS

app = FastAPI(title="WAN2.2 I2V Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 편의. 배포시 도메인 한정
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateResp(BaseModel):
    job_id: str

@app.post("/api/generate", response_model=GenerateResp)
async def generate(
    prompt: str = Form(...),
    duration: float = Form(4.0),
    fps: int = Form(16),
    width: int = Form(1280),
    height: int = Form(704),
    negative_prompt: str | None = Form(None),
    image: UploadFile | None = None,
):
    img_bytes = await image.read() if image is not None else None
    job_id = start_job(
        prompt=prompt,
        image_bytes=img_bytes,
        width=width,
        height=height,
        fps=fps,
        duration_sec=duration,
        negative_prompt=negative_prompt,
    )
    return GenerateResp(job_id=job_id)

@app.get("/api/status/{job_id}")
def status(job_id: str):
    info = JOBS.get(job_id)
    if not info:
        raise HTTPException(404, "job not found")
    resp = {
        "status": info["status"],
        "progress": info.get("progress", 0.0),
        "result": None,
        "error": info.get("error"),
        "error_code": info.get("error_code"),   # ← 추가
    }
    if info["status"] == "done" and info.get("result_path"):
        fname = os.path.basename(info["result_path"])
        resp["result"] = f"/api/video/{fname}"
    return JSONResponse(resp)

# --- Server-Sent Events: 진행률/에러를 푸시로 전달 ---
@app.get("/api/stream/{job_id}")
async def stream(job_id: str):
    async def event_gen():
        last_payload = None
        while True:
            info = JOBS.get(job_id)
            if not info:
                # 존재하지 않으면 스트림 종료
                yield f"event: update\ndata: {json.dumps({'status':'gone','progress':0})}\n\n"
                return

            payload = {
                "status": info["status"],
                "progress": info.get("progress", 0.0),
                "result": None,
                "error": info.get("error"),
                "error_code": info.get("error_code"),
            }
            if info["status"] == "done" and info.get("result_path"):
                fname = os.path.basename(info["result_path"])
                payload["result"] = f"/api/video/{fname}"

            # 변화가 있을 때만 보냄(트래픽 절약)
            if payload != last_payload:
                yield f"event: update\ndata: {json.dumps(payload)}\n\n"
                last_payload = payload

            # 완료/에러면 스트림 종료
            if info["status"] in ("done", "error"):
                return
            await asyncio.sleep(1.0)  # 푸시 주기

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_gen(), media_type="text/event-stream", headers=headers)

@app.get("/api/video/{filename}")
def get_video(filename: str):
    path = os.path.join("downloads", filename)
    if not os.path.exists(path):
        raise HTTPException(404, "file not found")
    return FileResponse(path, media_type="video/mp4", filename=filename)
