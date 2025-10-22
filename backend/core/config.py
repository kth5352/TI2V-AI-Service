# backend/core/config.py
"""
글로벌 설정/상수: 단일 소스
"""
import os
import torch

# 결과물 저장 디렉토리
MEDIA_DIR = os.getenv("MEDIA_DIR", "downloads")

# 모델 경로(로컬 권장)
MODEL_ID = os.path.join(os.path.dirname(__file__), "..", "models", "Wan2.2-TI2V-5B-Diffusers")

# 디바이스/데이터타입
HAS_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if HAS_CUDA else "cpu"
DTYPE = torch.float16 if HAS_CUDA else torch.float32

if HAS_CUDA:
    torch.backends.cuda.matmul.allow_tf32 = True
if HAS_CUDA and torch.backends.cudnn.is_available():
    torch.backends.cudnn.benchmark = True

# Windows에서 xFormers 멈춤 이슈 회피
USE_XFORMERS = (os.name != "nt")

# 세그먼트/프레임 규약
SEG_LEN_PREF = 25
OVERLAP = 6
MAX_FRAMES_PER_CALL = 121

# I2V 안정화
USE_ANCHOR_EVERY_SEG = True
USE_BLEND = False
ALPHA_ORIG = 0.7
