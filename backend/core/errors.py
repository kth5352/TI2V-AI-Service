# backend/core/errors.py
"""
에러 분류/실패 처리 — 상태/코드 일관성 유지
"""
from __future__ import annotations
import logging
import torch
from core.jobs import JOBS

log = logging.getLogger("wan-i2v")


def classify_error(exc: Exception) -> str:
    msg = (str(exc) or "").lower()
    if isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in msg:
        return "OUT_OF_MEMORY"
    if "hf_transfer" in msg:
        return "HF_TRANSFER_ERROR"
    if "does not appear to have a file named" in msg or "file not found" in msg or "no such file" in msg:
        return "MODEL_FILES_MISSING"
    if "divisible by 4" in msg or "num_frames" in msg or "invalid argument" in msg:
        return "INVALID_ARGUMENT"
    if "unexpected keyword argument 'callback'" in msg:
        return "UNSUPPORTED_CALLBACK_PARAM"
    if "unexpected keyword argument 'callback_steps'" in msg:
        return "UNSUPPORTED_CALLBACK_STEPS"
    if "frame extraction failed" in msg:
        return "FRAME_EXTRACTION_ERROR"
    if "truncated" in msg or "cannot identify image file" in msg or "image file is truncated" in msg:
        return "INVALID_IMAGE"
    return "RUNTIME_ERROR"


def fail_job(job_id: str, exc: Exception):
    code = classify_error(exc)
    JOBS[job_id]["status"] = "error"
    JOBS[job_id]["error"] = str(exc)
    JOBS[job_id]["error_code"] = code
    log.exception(f"[JOB {job_id[:8]}] ERROR [{code}]: {exc}")
