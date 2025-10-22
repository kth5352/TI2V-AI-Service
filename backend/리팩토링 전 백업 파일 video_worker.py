# backend/video_worker.py
"""
비즈니스 로직(파이프라인 제어/프레임 합성/인코딩/진행률/에러코드 표준화)

리팩토링 포인트
1) 상수/설정/전역자원 명확화: 모델경로, 디바이스, 파이프라인 싱글턴, 락
2) 세그먼트 플래닝 + 오버랩 크로스페이드 모듈화
3) 파이프라인 출력 포맷 다양성 흡수(_extract_frames)
4) 진행률 규약: 0~95% 디노이즈, 96~100% 인코딩
5) 에러코드 표준화(_classify_error) + 로깅 일관화(_fail_job)
6) JPEG EOI 보정/EXIF 회전보정 등 업로드 견고성 향상
"""

from __future__ import annotations

import io
import os
import uuid
import time
import threading
import logging
import inspect
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import torch
from PIL import Image as PILImage
from PIL import ImageFile, ImageOps

# 트렁케이트(미완료) JPEG 로딩 허용
ImageFile.LOAD_TRUNCATED_IMAGES = True

# (선택) GPU 사용률 표기 (없어도 동작)
try:
    import pynvml  # pip install nvidia-ml-py
    pynvml.nvmlInit()
    _nvml = True
except Exception:
    _nvml = False

# 비디오 인코딩
# pip install imageio imageio-ffmpeg
import imageio.v2 as imageio

# diffusers 파이프라인
from diffusers import (
    WanPipeline,                 # Text-to-Video
    WanImageToVideoPipeline,     # Image-to-Video
    UniPCMultistepScheduler,
)
try:
    from diffusers import AutoencoderKLWan
except Exception:
    AutoencoderKLWan = None

# ---------------------------------------------------------------------
# 로깅
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("wan-i2v")

# ---------------------------------------------------------------------
# 전역 상태/설정
# ---------------------------------------------------------------------
JOBS: Dict[str, Dict[str, Any]] = {}              # 작업 상태 저장소 (경량 KV)
JOBS_LOCK = threading.Lock()                      # 상태 갱신 락(필요 시 사용)

# 결과 파일 저장 디렉토리(외부에서 접근할 수 있도록 main.py와 합의)
MEDIA_DIR = os.getenv("MEDIA_DIR", "downloads")

# 모델: 로컬 프리트레인 경로(사전 다운로드 권장)
MODEL_ID = os.path.join(os.path.dirname(__file__), "models", "Wan2.2-TI2V-5B-Diffusers")
# 원격 캐시 사용 시:
# MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

_HAS_CUDA = torch.cuda.is_available()
_DEVICE = "cuda" if _HAS_CUDA else "cpu"
_DTYPE = torch.float16 if _HAS_CUDA else torch.float32

if _HAS_CUDA:
    torch.backends.cuda.matmul.allow_tf32 = True
if _HAS_CUDA and torch.backends.cudnn.is_available():
    torch.backends.cudnn.benchmark = True

# Windows xFormers 이슈 회피(기본 off)
_USE_XFORMERS = (os.name != "nt")

# 파이프라인 싱글턴
_PIPE_T2V = None
_PIPE_I2V = None
_PIPE_LOCK = threading.Lock()
_WARMED_T2V = False
_WARMED_I2V = False

# Pillow 리샘플링 상수 호환
try:
    RESAMPLE_BICUBIC = PILImage.Resampling.BICUBIC
except Exception:
    RESAMPLE_BICUBIC = PILImage.BICUBIC

# ---------------------------------------------------------------------
# 튜닝 파라미터
# ---------------------------------------------------------------------
SEG_LEN_PREF = 25           # 세그먼트 길이 기본값 (num_frames 규약 만족: (n-1)%4==0)
OVERLAP = 6                 # 세그 경계 크로스페이드 프레임 수
MAX_FRAMES_PER_CALL = 121   # WAN 제약 안전값

# I2V 안정화 옵션
USE_ANCHOR_EVERY_SEG = True  # 세그 시작마다 원본 이미지로 리셋
USE_BLEND = False            # 리셋 대신 블렌드 사용 시 True
ALPHA_ORIG = 0.7             # 블렌드 시 원본 비중 (0~1)

# ---------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------
def _gpu_str() -> str:
    """GPU/메모리 상태 문자열(로그 표시용)"""
    if not _HAS_CUDA:
        return "cpu"
    name = torch.cuda.get_device_name(0)
    try:
        free, total = torch.cuda.mem_get_info()
        used_gib = (total - free) / (1024 ** 3)
        total_gib = total / (1024 ** 3)
        mem = f"mem {used_gib:.2f}GiB/{total_gib:.2f}GiB used"
    except Exception:
        alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        resv = torch.cuda.memory_reserved() / (1024 ** 3)
        mem = f"alloc {alloc:.2f}GiB / reserved {resv:.2f}GiB"
    util = ""
    if _nvml:
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            u = pynvml.nvmlDeviceGetUtilizationRates(h)
            util = f" | util {u.gpu}% / mem {u.memory}%"
        except Exception:
            pass
    return f"{name} | {mem}{util}"


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


def _pil_from_bytes(b: bytes) -> PILImage.Image:
    """트렁케이트/EXIF 회전 포함, 최대한 복구 후 RGB 변환"""
    if not b:
        raise ValueError("Empty image bytes.")
    bb = _maybe_fix_truncated_jpeg(b)
    bio = io.BytesIO(bb)
    try:
        img = PILImage.open(bio)
        img.load()
    except Exception as e1:
        log.warning(f"[IMAGE] first load failed: {e1}. Retrying tolerant loader.")
        bio.seek(0)
        img = PILImage.open(bio)
        img.load()
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    return img.convert("RGB")


def _load_vae(model_id: str):
    """Wan 전용 VAE가 있다면 로드"""
    if AutoencoderKLWan is not None:
        return AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    return None


def _snap_hw(width: int, height: int, pipe) -> Tuple[int, int]:
    """
    모델 패치/스케일 배수에 맞춰 해상도 스냅.
    실패 시 32배수 fall-back. (반환은 (H, W))
    """
    try:
        mod_h = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[0]
        mod_w = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
        if not isinstance(mod_h, int) or not isinstance(mod_w, int):
            raise ValueError
    except Exception:
        mod_h, mod_w = 32, 32
    h = max(mod_h, (height // mod_h) * mod_h)
    w = max(mod_w, (width  // mod_w) * mod_w)
    return int(h), int(w)


def _nearest_valid_frames(n: int) -> int:
    """WAN 규약: (num_frames-1)%4 == 0 만족하도록 보정"""
    if (n - 1) % 4 == 0:
        return n
    r = int(round((n - 1) / 4.0) * 4 + 1)
    return max(9, r)


def _budget_megapixels(w: int, h: int, frames: int) -> float:
    return (w * h * max(1, frames)) / 1_000_000.0


def _autoscale_for_vram(
    w: int, h: int, frames: int, steps: int, vram_gb: float | None = None
) -> Tuple[int, int, int, int]:
    """
    해상도/프레임은 입력을 존중하되, 프레임 규칙만 보정.
    스텝은 28로 고정(현실적인 품질/시간 타협).
    """
    frames = _nearest_valid_frames(frames)
    steps = 28
    return w, h, frames, steps

# ---------------------------------------------------------------------
# 세그먼트 플래너 & 크로스페이드
# ---------------------------------------------------------------------
def _plan_segments(total_frames: int, seg_len_pref: int = SEG_LEN_PREF, overlap: int = OVERLAP) -> List[int]:
    """
    총 프레임을 세그먼트 리스트로 분할한다.
    - 첫 세그: 오버랩 없음
    - 이후 세그: 'overlap' 길이만큼 앞 세그와 겹치며, (n-1)%4==0 규칙 유지
    - 최종 합성 결과 길이가 정확히 total_frames가 되도록 미세 조정
    """
    if total_frames <= 0:
        return []
    segs: List[int] = []
    eff_acc = 0  # 오버랩 제외 유효 누적

    pref = _nearest_valid_frames(max(9, seg_len_pref))
    while eff_acc < total_frames:
        if not segs:
            need = total_frames - eff_acc
            L = min(pref, need if need <= MAX_FRAMES_PER_CALL else pref)
            L = _nearest_valid_frames(max(9, min(MAX_FRAMES_PER_CALL, L)))
            segs.append(L)
            eff_acc += L
        else:
            need = total_frames - eff_acc
            L_eff_target = min(pref - overlap, need)  # 유효 기여 목표
            L = _nearest_valid_frames(max(9, min(MAX_FRAMES_PER_CALL, L_eff_target + overlap)))
            if L - overlap <= 0:
                L = _nearest_valid_frames(max(9, overlap + 1))
            segs.append(L)
            eff_acc += max(0, L - overlap)

    # 미세 오차 조정
    eff_total = segs[0] + sum(max(0, L - overlap) for L in segs[1:])
    if eff_total != total_frames and len(segs) >= 1:
        delta = eff_total - total_frames
        if delta > 0:
            L_last = segs[-1]
            if len(segs) == 1:
                target = max(9, L_last - delta)
                segs[-1] = _nearest_valid_frames(target)
            else:
                target_eff = max(1, (L_last - overlap) - delta)
                segs[-1] = _nearest_valid_frames(target_eff + overlap)

    return segs


def _crossfade_frames(tail: List[np.ndarray], head: List[np.ndarray], overlap: int) -> List[np.ndarray]:
    """
    두 시퀀스 사이를 overlap 길이로 프레임별 알파블렌딩.
    tail[-overlap:], head[:overlap] 각각 같은 길이로 맞춰 선형 가중 합성
    """
    if overlap <= 0:
        return []
    n = min(overlap, len(tail), len(head))
    if n <= 0:
        return []
    out = []
    for i in range(n):
        a = (i + 1) / (n + 1)  # 0→1 (끝쪽이 head 비중 ↑)
        t = tail[-n + i].astype(np.float32)
        h = head[i].astype(np.float32)
        b = (1.0 - a) * t + a * h
        out.append(np.clip(b, 0, 255).astype(np.uint8))
    return out

# ---------------------------------------------------------------------
# 파이프라인 로딩/워밍업
# ---------------------------------------------------------------------
def _apply_memory_savers(pipe, *, is_i2v: bool):
    """메모리 절약/성능 옵션 적용 (가능한 경우에만)"""
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    if not is_i2v:
        try:
            pipe.enable_vae_tiling()
        except Exception:
            pass
    if _USE_XFORMERS:
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass


def _warmup_once(pipe, is_i2v: bool):
    """첫 호출 오래 걸리는 현상 완화용 워밍업"""
    global _WARMED_T2V, _WARMED_I2V
    if (is_i2v and _WARMED_I2V) or ((not is_i2v) and _WARMED_T2V):
        return
    try:
        log.info("[WARMUP] tiny warmup…")
        with torch.inference_mode():
            ctx = torch.autocast("cuda", dtype=_DTYPE) if _HAS_CUDA else nullcontext()
            with ctx:
                try:
                    pipe.set_progress_bar_config(disable=True)
                except Exception:
                    pass
                kwargs = dict(prompt="warmup", height=256, width=256, num_frames=5, num_inference_steps=1, guidance_scale=2.5)
                if is_i2v:
                    kwargs["image"] = PILImage.new("RGB", (320, 320), (0, 0, 0))
                pipe(**kwargs)
        log.info("[WARMUP] done.")
    except Exception as e:
        log.warning(f"[WARMUP] skipped: {e}")
    if is_i2v:
        _WARMED_I2V = True
    else:
        _WARMED_T2V = True


def get_t2v_pipe():
    """Text→Video 파이프라인 싱글턴 획득"""
    global _PIPE_T2V
    if _PIPE_T2V is None:
        with _PIPE_LOCK:
            if _PIPE_T2V is None:
                vae = _load_vae(MODEL_ID)
                pipe = WanPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=_DTYPE, local_files_only=True)
                pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
                _apply_memory_savers(pipe, is_i2v=False)
                try:
                    pipe.enable_sequential_cpu_offload()
                    log.info("[PIPE] T2V enabled CPU Offload.")
                except Exception:
                    pipe.to(_DEVICE)
                _PIPE_T2V = pipe
                log.info(f"[PIPE] T2V loaded on {_DEVICE} ({_gpu_str()}); dtype={_DTYPE}")
                _warmup_once(_PIPE_T2V, is_i2v=False)
    return _PIPE_T2V


def get_i2v_pipe():
    """Image→Video 파이프라인 싱글턴 획득"""
    global _PIPE_I2V
    if _PIPE_I2V is None:
        with _PIPE_LOCK:
            if _PIPE_I2V is None:
                pipe = WanImageToVideoPipeline.from_pretrained(MODEL_ID, torch_dtype=_DTYPE, local_files_only=True)
                pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
                _apply_memory_savers(pipe, is_i2v=True)
                try:
                    pipe.enable_model_cpu_offload()
                    log.info("[PIPE] I2V enabled MODEL CPU Offload.")
                except Exception:
                    pipe.to(_DEVICE)
                try:
                    pipe.vae.to(_DEVICE)
                    log.info("[PIPE] I2V VAE pinned on GPU.")
                except Exception:
                    pass
                log.info(f"[PIPE] I2V loaded on {_DEVICE} ({_gpu_str()}); dtype={_DTYPE}")
                _warmup_once(pipe, is_i2v=True)
                _PIPE_I2V = pipe
    return _PIPE_I2V

# ---------------------------------------------------------------------
# 출력 정규화 (파이프라인 버전에 따라 달라질 수 있는 인터페이스 흡수)
# ---------------------------------------------------------------------
def _to_uint8_hwc(arr: np.ndarray) -> np.ndarray:
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


def _extract_frames(job_id: str, out_obj) -> List[np.ndarray]:
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
        log.info(f"[JOB {job_id[:8]}] Frame source: out.frames")
        return _normalize_frame_sequence(job_id, seq)
    if hasattr(out_obj, "videos"):
        vids = getattr(out_obj, "videos")
        if hasattr(vids, "cpu"):
            vids = vids.detach().cpu().numpy()
        if isinstance(vids, np.ndarray) and vids.ndim == 5:
            if vids.shape[2] in (1, 3):
                seq = np.transpose(vids[0], (0, 2, 3, 1))
            else:
                seq = vids[0]
            log.info(f"[JOB {job_id[:8]}] Frame source: out.videos (B={vids.shape[0]},F={vids.shape[1]})")
            return [_to_uint8_hwc(fr) for fr in seq]
        raise AttributeError("Unsupported 'videos' format.")
    if isinstance(out_obj, (list, tuple)):
        if len(out_obj) == 0:
            raise AttributeError("Pipeline returned empty list/tuple.")
        log.info(f"[JOB {job_id[:8]}] Frame source: out[0] (list/tuple)")
        return _normalize_frame_sequence(job_id, out_obj[0])
    if isinstance(out_obj, dict):
        for k in ("frames", "videos"):
            if k in out_obj:
                return _extract_frames(job_id, SimpleNamespace(**out_obj))
        raise AttributeError("Dict output without 'frames' or 'videos'.")
    raise AttributeError(f"Unexpected pipeline output type: {type(out_obj)}")


def _normalize_frame_sequence(job_id: str, seq) -> List[np.ndarray]:
    """PIL/np/tensor 혼재를 [np.uint8 HWC] 리스트로 표준화"""
    frames_u8: List[np.ndarray] = []
    if (hasattr(seq, "cpu") and not isinstance(seq, (list, tuple))) or isinstance(seq, np.ndarray):
        arr = seq.detach().cpu().numpy() if hasattr(seq, "cpu") else seq
        if arr.ndim == 5:
            if arr.shape[2] in (1, 3):
                arr = np.transpose(arr[0], (0, 2, 3, 1))
            else:
                arr = arr[0]
            return [_to_uint8_hwc(fr) for fr in arr]
        if arr.ndim == 4:
            if arr.shape[1] in (1, 3):
                arr = np.transpose(arr, (0, 2, 3, 1))
            return [_to_uint8_hwc(fr) for fr in arr]
        if arr.ndim == 3:
            return [_to_uint8_hwc(arr)]
    if isinstance(seq, (list, tuple)):
        for idx, fr in enumerate(seq):
            if hasattr(fr, "cpu"):
                fr = fr.detach().cpu().numpy()
            if isinstance(fr, PILImage.Image):
                fr = np.array(fr)
            if not isinstance(fr, np.ndarray):
                raise TypeError(f"Unsupported frame item at {idx}: {type(fr)}")
            frames_u8.append(_to_uint8_hwc(fr))
        return frames_u8
    raise TypeError(f"Unsupported frame sequence type: {type(seq)}")

# ---------------------------------------------------------------------
# 비디오 인코딩
# ---------------------------------------------------------------------
def _export_video(job_id: str, frames: List[np.ndarray], out_path: str, fps: int):
    """96%→100%: imageio-ffmpeg로 H.264 인코딩(프로그레스 반영)"""
    log.info(f"[JOB {job_id[:8]}] encoding video -> {out_path}")
    JOBS[job_id]["progress"] = 0.96
    if not frames:
        raise RuntimeError("No frames to encode.")
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", ffmpeg_params=["-crf", "18", "-preset", "medium"])
    try:
        total = len(frames)
        for i, arr in enumerate(frames, 1):
            writer.append_data(arr)
            JOBS[job_id]["progress"] = round(0.96 + 0.04 * (i / total), 4)
            # 10% 단위로만 로그
            if total >= 10 and (i % max(1, total // 10) == 0):
                log.info(f"[JOB {job_id[:8]}] encoding {i}/{total} frames ({100*i/total:.1f}%)")
    finally:
        writer.close()
    log.info(f"[JOB {job_id[:8]}] export done: {out_path}")

# ---------------------------------------------------------------------
# 에러 코드 표준화
# ---------------------------------------------------------------------
def _classify_error(exc: Exception) -> str:
    """예외 메시지를 표준 에러코드로 매핑"""
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


def _fail_job(job_id: str, exc: Exception):
    """상태 저장 + 표준코드 + 스택로깅"""
    code = _classify_error(exc)
    JOBS[job_id]["status"] = "error"
    JOBS[job_id]["error"] = str(exc)
    JOBS[job_id]["error_code"] = code
    log.exception(f"[JOB {job_id[:8]}] ERROR [{code}]: {exc}")

# ---------------------------------------------------------------------
# 퍼블릭 API: 잡 시작 (main.py에서 import)
# ---------------------------------------------------------------------
def start_job(
    prompt: str,
    image_bytes: Optional[bytes],
    width: int,
    height: int,
    fps: int,
    duration_sec: float,
    negative_prompt: Optional[str] = None,
    guidance_scale: float = 5.0,
    num_inference_steps: int = 28,
    download_dir: str = MEDIA_DIR,
) -> str:
    """
    새 비디오 생성 잡을 백그라운드 스레드로 실행.
    - image_bytes 존재 → I2V, 없으면 T2V
    - 진행률: 0~0.95(디노이즈) / 0.96~1.0(인코딩)
    - 완료 시: result_path 설정, status=done
    - 에러 시: status=error, error/error_code 설정
    """
    os.makedirs(download_dir, exist_ok=True)

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "status": "queued",
        "progress": 0.0,
        "result_path": None,
        "error": None,
        "error_code": None,
        "meta": {"fps": fps, "duration": duration_sec, "size": (width, height)},
    }

    def run():
        t0 = time.time()
        try:
            JOBS[job_id]["status"] = "running"
            is_i2v = image_bytes is not None
            pipe = get_i2v_pipe() if is_i2v else get_t2v_pipe()

            # 총 프레임 정확 계산
            total_frames_req = max(1, int(round(fps * duration_sec)))

            # 규약 보정(프레임) + 스텝 28 고정
            steps_local = num_inference_steps
            w0, h0, frames_ruled, steps_fixed = _autoscale_for_vram(width, height, total_frames_req, steps_local, vram_gb=None)
            h, w = _snap_hw(w0, h0, pipe)  # (H,W)
            steps_local = steps_fixed

            # 세그먼트 플랜 수립
            seg_plans = _plan_segments(frames_ruled, SEG_LEN_PREF, OVERLAP)
            num_segments = len(seg_plans)
            total_denoise_steps = max(1, steps_local) * max(1, num_segments)

            plan_mp = _budget_megapixels(w, h, min(MAX_FRAMES_PER_CALL, SEG_LEN_PREF))
            log.info(
                f"[JOB {job_id[:8]}] mode={'I2V' if is_i2v else 'T2V'} | "
                f"req={width}x{height}@{total_frames_req}f,{fps}fps, steps={num_inference_steps} → "
                f"plan: {num_segments} segments (~{SEG_LEN_PREF}f, ov={OVERLAP}) @ {w}x{h}, "
                f"per-call steps={steps_local} (~{plan_mp:.1f} MP/call) | "
                f"gs={guidance_scale} | device={_DEVICE} ({_gpu_str()})"
            )

            JOBS[job_id]["progress"] = 0.05

            # 입력 이미지(I2V)
            original_img = _pil_from_bytes(image_bytes) if is_i2v else None
            current_img = None
            if original_img is not None:
                original_img = original_img.resize((w, h), RESAMPLE_BICUBIC)
                current_img = original_img.copy()

            # 결과 프레임 버퍼
            all_frames: List[np.ndarray] = []
            denoise_step_counter = 0  # 전체 스텝 카운터(진행률 계산용)

            # 세그먼트별 on_step_end 콜백(전역 진행률 반영)
            def make_on_step_end(seg_index: int):
                def cb_on_step_end(pipe_, step: int, timestep, callback_kwargs):
                    nonlocal denoise_step_counter
                    done_steps = seg_index * max(1, steps_local) + (step + 1)
                    denoise_step_counter = done_steps
                    denoise_pct = min(0.95, denoise_step_counter / float(max(1, total_denoise_steps)))
                    JOBS[job_id]["progress"] = round(max(JOBS[job_id]["progress"], denoise_pct), 4)
                    return callback_kwargs
                return cb_on_step_end

            # ===== 세그먼트 실행 =====
            for si, seg_len in enumerate(seg_plans):
                # I2V: 세그 시작 시 앵커 리셋 또는 블렌드
                if is_i2v and original_img is not None:
                    if USE_ANCHOR_EVERY_SEG:
                        current_img = original_img.copy()
                    elif USE_BLEND and current_img is not None:
                        try:
                            current_img = PILImage.blend(current_img, original_img, ALPHA_ORIG)
                        except Exception:
                            current_img = original_img.copy()
                    elif current_img is None:
                        current_img = original_img.copy()

                # 파이프 호출
                with torch.inference_mode():
                    ctx = torch.autocast("cuda", dtype=_DTYPE) if _HAS_CUDA else nullcontext()
                    with ctx:
                        sig_params = set(inspect.signature(pipe.__call__).parameters.keys())
                        call_kwargs = dict(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            height=h,
                            width=w,
                            num_frames=seg_len,
                            num_inference_steps=steps_local,
                            guidance_scale=guidance_scale,
                        )
                        if is_i2v and "image" in sig_params:
                            call_kwargs["image"] = current_img
                        if "callback_on_step_end" in sig_params:
                            call_kwargs["callback_on_step_end"] = make_on_step_end(si)
                        try:
                            pipe.set_progress_bar_config(disable=True)
                        except Exception:
                            pass
                        out = pipe(**call_kwargs)

                # 프레임 추출
                seg_frames = _extract_frames(job_id, out)
                if not seg_frames:
                    raise RuntimeError(f"Segment {si} returned no frames.")

                # 다음 세그 준비용 current_img 업데이트
                if is_i2v:
                    try:
                        current_img = PILImage.fromarray(seg_frames[-1])
                    except Exception:
                        current_img = original_img.copy()

                # 합성(오버랩 크로스페이드)
                if si == 0:
                    all_frames.extend(seg_frames)
                else:
                    tail = all_frames[-OVERLAP:] if len(all_frames) >= OVERLAP else all_frames[-1:]
                    head = seg_frames[:OVERLAP] if len(seg_frames) >= OVERLAP else seg_frames[:1]
                    blended = _crossfade_frames(tail, head, OVERLAP)
                    if OVERLAP > 0 and len(all_frames) >= len(tail):
                        all_frames = all_frames[:-len(tail)]
                    all_frames.extend(blended)
                    all_frames.extend(seg_frames[OVERLAP:])

            # 총 길이 정확화(이론상 정확, 안전 보정)
            total_frames_req = max(1, int(round(fps * duration_sec)))
            if len(all_frames) > total_frames_req:
                all_frames = all_frames[:total_frames_req]
            elif len(all_frames) < total_frames_req and all_frames:
                last = all_frames[-1]
                all_frames.extend([last.copy() for _ in range(total_frames_req - len(all_frames))])

            log.info(f"[JOB {job_id[:8]}] Denoising complete. Collected {len(all_frames)} frames (target {total_frames_req}).")

            # ===== 인코딩 =====
            out_path = os.path.join(download_dir, f"wan2_{job_id}.mp4")
            try:
                _export_video(job_id, all_frames, out_path, fps=fps)
            except Exception as e:
                _fail_job(job_id, e)
                return

            JOBS[job_id].update(status="done", progress=1.0, result_path=out_path)
            log.info(f"[JOB {job_id[:8]}] DONE in {time.time()-t0:.1f}s")

        except Exception as e:
            _fail_job(job_id, e)

    threading.Thread(target=run, daemon=True).start()
    return job_id
