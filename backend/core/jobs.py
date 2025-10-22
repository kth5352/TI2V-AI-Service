# backend/core/jobs.py
"""
전역 잡 상태 저장소 — 단순 딕셔너리 + (선택) 락
"""
from __future__ import annotations
from typing import Dict, Any
from threading import Lock

JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = Lock()
