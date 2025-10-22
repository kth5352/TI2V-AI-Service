# backend/utils/encode.py
"""
비디오 인코딩 — 진행률 0.96→1.0 반영
"""
from __future__ import annotations
from typing import List
import imageio.v2 as imageio
import numpy as np
import logging
from core.jobs import JOBS

log = logging.getLogger("wan-i2v")


def export_video(job_id: str, frames: List[np.ndarray], out_path: str, fps: int):
    """imageio-ffmpeg(H.264)로 인코딩하면서 진행률 갱신"""
    JOBS[job_id]["progress"] = 0.96
    if not frames:
        raise RuntimeError("No frames to encode.")
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", ffmpeg_params=["-crf", "18", "-preset", "medium"])
    try:
        total = len(frames)
        for i, arr in enumerate(frames, 1):
            writer.append_data(arr)
            JOBS[job_id]["progress"] = round(0.96 + 0.04 * (i / total), 4)
            if total >= 10 and (i % max(1, total // 10) == 0):
                log.info(f"[JOB {job_id[:8]}] encoding {i}/{total} frames ({100*i/total:.1f}%)")
    finally:
        writer.close()
