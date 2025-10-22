# backend/utils/frames.py
"""
파이프라인 출력 → 표준 프레임 리스트(uint8 HWC) 변환
"""
from __future__ import annotations
from typing import List
import numpy as np
from PIL import Image as PILImage
from types import SimpleNamespace
import logging

log = logging.getLogger("wan-i2v")


def to_uint8_hwc(arr: np.ndarray) -> np.ndarray:
    """값 범위/채널순서를 uint8 HWC로 표준화"""
    if arr.dtype != np.uint8:
        a = arr.astype(np.float32)
        if a.max() <= 1.0 + 1e-3 and a.min() >= 0.0 - 1e-3:
            a = a * 255.0
        elif a.min() < 0.0 and a.max() <= 1.0 + 1e-3:
            a = (a + 1.0) * 127.5  # [-1,1] → [0,255]
        a = np.clip(a, 0, 255).astype(np.uint8)
    else:
        a = arr
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    return a


def normalize_frame_sequence(job_id: str, seq) -> List[np.ndarray]:
    """PIL/np/tensor 혼재를 [np.uint8 HWC] 리스트로 표준화"""
    frames_u8: List[np.ndarray] = []
    if (hasattr(seq, "cpu") and not isinstance(seq, (list, tuple))) or isinstance(seq, np.ndarray):
        arr = seq.detach().cpu().numpy() if hasattr(seq, "cpu") else seq
        if arr.ndim == 5:
            if arr.shape[2] in (1, 3):
                arr = np.transpose(arr[0], (0, 2, 3, 1))
            else:
                arr = arr[0]
            return [to_uint8_hwc(fr) for fr in arr]
        if arr.ndim == 4:
            if arr.shape[1] in (1, 3):
                arr = np.transpose(arr, (0, 2, 3, 1))
            return [to_uint8_hwc(fr) for fr in arr]
        if arr.ndim == 3:
            return [to_uint8_hwc(arr)]
    if isinstance(seq, (list, tuple)):
        for idx, fr in enumerate(seq):
            if hasattr(fr, "cpu"):
                fr = fr.detach().cpu().numpy()
            if isinstance(fr, PILImage.Image):
                fr = np.array(fr)
            if not isinstance(fr, np.ndarray):
                raise TypeError(f"Unsupported frame item at {idx}: {type(fr)}")
            frames_u8.append(to_uint8_hwc(fr))
        return frames_u8
    raise TypeError(f"Unsupported frame sequence type: {type(seq)}")


def extract_frames(job_id: str, out_obj) -> List[np.ndarray]:
    """
    Diffusers 출력의 다양한 케이스를 흡수:
    - out.frames[0] (List[PIL/np/tensor])
    - out.videos (np/tensor: (B,F,C,H,W) or (B,F,H,W,C))
    - list/tuple/dict 등
    """
    log.info(f"[JOB {job_id[:8]}] Output type: {type(out_obj)}")
    if hasattr(out_obj, "frames"):
        frames = getattr(out_obj, "frames")
        seq = frames[0] if isinstance(frames, (list, tuple)) and len(frames) > 0 else frames
        return normalize_frame_sequence(job_id, seq)
    if hasattr(out_obj, "videos"):
        vids = getattr(out_obj, "videos")
        if hasattr(vids, "cpu"):
            vids = vids.detach().cpu().numpy()
        import numpy as np
        if isinstance(vids, np.ndarray) and vids.ndim == 5:
            if vids.shape[2] in (1, 3):
                seq = np.transpose(vids[0], (0, 2, 3, 1))
            else:
                seq = vids[0]
            return [to_uint8_hwc(fr) for fr in seq]
        raise AttributeError("Unsupported 'videos' format.")
    if isinstance(out_obj, (list, tuple)):
        if len(out_obj) == 0:
            raise AttributeError("Pipeline returned empty list/tuple.")
        return normalize_frame_sequence(job_id, out_obj[0])
    if isinstance(out_obj, dict):
        for k in ("frames", "videos"):
            if k in out_obj:
                return extract_frames(job_id, SimpleNamespace(**out_obj))
        raise AttributeError("Dict output without 'frames' or 'videos'.")
    raise AttributeError(f"Unexpected pipeline output type: {type(out_obj)}")
