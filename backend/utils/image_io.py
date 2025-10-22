# backend/utils/image_io.py
"""
이미지 로딩/복구/리사이즈 헬퍼
"""
from __future__ import annotations

import io
import logging
from PIL import Image as PILImage, ImageFile, ImageOps

log = logging.getLogger("wan-i2v")

# 트렁케이트 JPEG 허용
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Pillow 리샘플링 상수 호환
try:
    RESAMPLE_BICUBIC = PILImage.Resampling.BICUBIC
except Exception:
    RESAMPLE_BICUBIC = PILImage.BICUBIC


def _maybe_fix_truncated_jpeg(b: bytes) -> bytes:
    """EOI(0xFFD9) 누락 JPEG 복구 시도"""
    if not b:
        return b
    try:
        if len(b) >= 2 and b[0] == 0xFF and b[1] == 0xD8 and not b.endswith(b"\xFF\xD9"):
            log.warning("[IMAGE] JPEG missing EOI, appending 0xFFD9.")
            return b + b"\xFF\xD9"
    except Exception:
        pass
    return b


def pil_from_bytes(b: bytes) -> PILImage.Image:
    """트렁케이트/EXIF 회전 포함, 최대한 복구 후 RGB 변환"""
    if not b:
        raise ValueError("Empty image bytes.")
    bb = _maybe_fix_truncated_jpeg(b)
    bio = io.BytesIO(bb)
    try:
        img = PILImage.open(bio); img.load()
    except Exception as e1:
        log.warning(f"[IMAGE] first load failed: {e1}. Retrying tolerant loader.")
        bio.seek(0)
        img = PILImage.open(bio); img.load()
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    return img.convert("RGB")
