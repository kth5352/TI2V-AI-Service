# backend/video_worker.py — WAN2.2 I2V/T2V
# 진행률(SSE) + 안전 워밍업 + 세그먼트(25f) + 6f 크로스페이드 + I2V 앵커 리셋
# + VRAM 안전 스냅 + 동적 콜백 호환 + 출력 안전 추출 + 인코딩 진행 표시 + 에러코드
import io
import os
import uuid
import time
import threading
import logging
import inspect
from typing import Optional, Dict, Any, Tuple, List
from contextlib import nullcontext
from types import SimpleNamespace

import torch
import numpy as np
from PIL import Image as PILImage
from PIL import ImageFile, ImageOps

# 트렁케이트 이미지 허용 (손상/미완료 업로드 허용)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# (선택) GPU 사용률 표기를 위해
try:
    import pynvml  # pip install nvidia-ml-py
    pynvml.nvmlInit()
    _nvml = True
except Exception:
    _nvml = False

# 인코딩 진행률 출력
# pip install imageio imageio-ffmpeg
import imageio.v2 as imageio

# ------------------------------
# 로깅
# ------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("wan-i2v")

# ------------------------------
# Diffusers
# ------------------------------
from diffusers import (
    WanPipeline,                 # Text-to-Video
    WanImageToVideoPipeline,     # Image-to-Video
    UniPCMultistepScheduler,
)
try:
    from diffusers import AutoencoderKLWan
except Exception:
    AutoencoderKLWan = None

# ------------------------------
# 글로벌 상태
# ------------------------------
JOBS: Dict[str, Dict[str, Any]] = {}

# 가능하면 로컬 모델 권장 (huggingface-cli로 미리 받아둔 경로)
MODEL_ID = os.path.join(os.path.dirname(__file__), "models", "Wan2.2-TI2V-5B-Diffusers")
# 원격 자동 캐시: MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

_has_cuda = torch.cuda.is_available()
_device = "cuda" if _has_cuda else "cpu"
_dtype = torch.float16 if _has_cuda else torch.float32

if _has_cuda:
    torch.backends.cuda.matmul.allow_tf32 = True
if _has_cuda and torch.backends.cudnn.is_available():
    torch.backends.cudnn.benchmark = True

# Windows에서 xFormers 멈춤 이슈가 있어 기본 OFF
_USE_XFORMERS = (os.name != "nt")

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

# ------------------------------
# 튜닝 파라미터 (필요시 조정)
# ------------------------------
SEG_LEN_PREF = 25     # 세그먼트 길이 기본값(권장). (n-1)%4==0 만족 (25->OK)
OVERLAP = 6           # 세그먼트 경계 크로스페이드 프레임 수
MAX_FRAMES_PER_CALL = 121  # WAN 제한 안전값

# I2V 안정화 옵션
USE_ANCHOR_EVERY_SEG = True  # 세그먼트 시작마다 원본으로 리셋(권장)
USE_BLEND = False            # 리셋 대신 원본/직전프레임 블렌드 사용 시 True
ALPHA_ORIG = 0.7             # 블렌드 시 원본 비중 (0~1)

# ------------------------------
# 유틸
# ------------------------------
def _gpu_str() -> str:
    if not _has_cuda:
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
    """EOI(0xFFD9)가 누락된 JPEG일 경우 끝에 붙여서 복구 시도."""
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
    """트렁케이트/EOI누락 이미지도 최대한 복구해 RGB로 반환."""
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
    if AutoencoderKLWan is not None:
        return AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    return None


def _snap_hw(width: int, height: int, pipe) -> Tuple[int, int]:
    """모델 패치/스케일 배수로 스냅 (실패 시 32배수). 반환 (H,W)."""
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
    """(num_frames-1)%4 == 0 만족하도록 보정."""
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
    해상도/프레임은 프론트 입력을 최대한 존중하되, 프레임 규칙만 보정.
    스텝은 요청대로 28 고정.
    """
    frames = _nearest_valid_frames(frames)
    steps = 28
    return w, h, frames, steps

# ------------------------------
# 세그먼트 플래너 & 크로스페이드
# ------------------------------
def _plan_segments(total_frames: int, seg_len_pref: int = SEG_LEN_PREF, overlap: int = OVERLAP) -> List[int]:
    """
    총 프레임(total_frames)을 세그먼트들로 쪼갠다.
    각 세그 길이는 (n-1)%4==0 만족. 첫 세그는 오버랩 없이, 이후 세그는 overlap만큼
    이전과 겹치며, 최종 합성 결과가 정확히 total_frames가 되도록 길이를 조정한다.
    """
    if total_frames <= 0:
        return []
    segs: List[int] = []
    eff_acc = 0  # 실제로 쌓이는 유효 프레임(오버랩 제외)

    pref = _nearest_valid_frames(max(9, seg_len_pref))
    while eff_acc < total_frames:
        if not segs:
            # 첫 세그: 전 오버랩 없음
            need = total_frames - eff_acc
            L = min(pref, need if need <= MAX_FRAMES_PER_CALL else pref)
            L = _nearest_valid_frames(max(9, min(MAX_FRAMES_PER_CALL, L)))
            segs.append(L)
            eff_acc += L
        else:
            # 이후 세그: 오버랩만큼은 실효 X
            need = total_frames - eff_acc
            # 이번 세그의 유효 기여 = L - overlap
            # need를 넘지 않도록 L을 정한다
            L_eff_target = min(pref - overlap, need)  # 유효 기여 목표
            L = L_eff_target + overlap
            # 규칙/상한 보정
            L = _nearest_valid_frames(max(9, min(MAX_FRAMES_PER_CALL, L)))
            # 혹시 유효 기여가 0 이하로 나오면 최소 길이로 강제
            if L - overlap <= 0:
                L = _nearest_valid_frames(max(9, overlap + 1))
            segs.append(L)
            eff_acc += max(0, L - overlap)

        # 마지막에 살짝 오버/언더가 생기면 다음 루프에서 보정됨

    # 마지막 세그가 너무 과한 경우(유효 기여가 total_frames를 크게 초과) 미세 조정
    # 실무상 거의 안 걸리지만 안전용.
    eff_total = segs[0] + sum(max(0, L - overlap) for L in segs[1:])
    if eff_total != total_frames:
        delta = eff_total - total_frames
        # delta>0이면 마지막 세그 길이를 줄여본다.
        if delta > 0 and len(segs) >= 1:
            L_last = segs[-1]
            # 마지막 세그 유효 기여는 (L_last - overlap) (첫 세그면 그냥 L_last)
            if len(segs) == 1:
                target = max(9, L_last - delta)
                segs[-1] = _nearest_valid_frames(target)
            else:
                target_eff = max(1, (L_last - overlap) - delta)
                segs[-1] = _nearest_valid_frames(target_eff + overlap)

    return segs


def _crossfade_frames(tail: List[np.ndarray], head: List[np.ndarray], overlap: int) -> List[np.ndarray]:
    """
    두 시퀀스 사이를 overlap 길이만큼 프레임별 알파블렌딩으로 연결.
    tail[-overlap:], head[:overlap]를 같은 길이로 맞춘 뒤 0→1 선형 가중으로 합성.
    """
    if overlap <= 0:
        return []
    n = min(overlap, len(tail), len(head))
    if n <= 0:
        return []
    out = []
    for i in range(n):
        a = (i + 1) / (n + 1)  # 0~1 사이(끝쪽이 head 쪽 비중)
        t = tail[-n + i].astype(np.float32)
        h = head[i].astype(np.float32)
        b = (1.0 - a) * t + a * h
        out.append(np.clip(b, 0, 255).astype(np.uint8))
    return out

# ------------------------------
# 파이프라인 로딩
# ------------------------------
def _apply_memory_savers(pipe, *, is_i2v: bool):
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
    """초소형 워밍업 1회로 첫 스텝 정지 방지."""
    global _WARMED_T2V, _WARMED_I2V
    if (is_i2v and _WARMED_I2V) or ((not is_i2v) and _WARMED_T2V):
        return
    try:
        log.info("[WARMUP] tiny warmup…")
        with torch.inference_mode():
            ctx = torch.autocast("cuda", dtype=_dtype) if _has_cuda else nullcontext()
            with ctx:
                try:
                    pipe.set_progress_bar_config(disable=True)
                except Exception:
                    pass
                kwargs = dict(prompt="warmup",
                              height=256, width=256,
                              num_frames=5,
                              num_inference_steps=1,
                              guidance_scale=2.5)
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
    global _PIPE_T2V
    if _PIPE_T2V is None:
        with _PIPE_LOCK:
            if _PIPE_T2V is None:
                vae = _load_vae(MODEL_ID)
                pipe = WanPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=_dtype, local_files_only=True)
                pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
                _apply_memory_savers(pipe, is_i2v=False)
                try:
                    pipe.enable_sequential_cpu_offload()
                    log.info("[PIPE] T2V enabled CPU Offload.")
                except Exception:
                    pipe.to(_device)
                _PIPE_T2V = pipe
                log.info(f"[PIPE] T2V loaded on {_device} ({_gpu_str()}); dtype={_dtype}")
                _warmup_once(_PIPE_T2V, is_i2v=False)
    return _PIPE_T2V


def get_i2v_pipe():
    global _PIPE_I2V
    if _PIPE_I2V is None:
        with _PIPE_LOCK:
            if _PIPE_I2V is None:
                pipe = WanImageToVideoPipeline.from_pretrained(
                    MODEL_ID, torch_dtype=_dtype, local_files_only=True
                )
                pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
                _apply_memory_savers(pipe, is_i2v=True)
                pipe.to(_device)
                try:
                    pipe.enable_model_cpu_offload()
                    log.info("[PIPE] I2V enabled MODEL CPU Offload.")
                except Exception:
                    pipe.to(_device)
                try:
                    pipe.vae.to(_device)
                    log.info("[PIPE] I2V VAE pinned on GPU.")
                except Exception:
                    pass
                log.info(f"[PIPE] I2V loaded on {_device} ({_gpu_str()}); dtype={_dtype}")
                _warmup_once(pipe, is_i2v=True)
                _PIPE_I2V = pipe
    return _PIPE_I2V

# ------------------------------
# 출력 → 프레임 리스트 표준화
# ------------------------------
def _to_uint8_hwc(arr: np.ndarray) -> np.ndarray:
    """다양한 범위/채널 순서를 안전하게 uint8 HWC로 변환."""
    if arr.dtype != np.uint8:
        a = arr.astype(np.float32)
        if a.max() <= 1.0 + 1e-3 and a.min() >= 0.0 - 1e-3:
            a = a * 255.0
        elif a.min() < 0.0 and a.max() <= 1.0 + 1e-3:
            a = (a + 1.0) * 127.5  # [-1,1]→[0,255]
        a = np.clip(a, 0, 255).astype(np.uint8)
    else:
        a = arr
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    return a


def _extract_frames(job_id: str, out_obj) -> List[np.ndarray]:
    """
    Diffusers 출력의 다양한 케이스를 흡수:
      - out.frames[0] (List[PIL/Image/np/tensor])
      - out.videos (np/tensor: (B,F,C,H,W) or (B,F,H,W,C))
      - list/tuple 등
    반환: [np.uint8(H,W,C), ...]
    """
    log.info(f"[JOB {job_id[:8]}] Output type: {type(out_obj)}")
    if hasattr(out_obj, "frames"):
        frames = getattr(out_obj, "frames")
        seq = frames[0] if isinstance(frames, (list, tuple)) and len(frames) > 0 else frames
        log.info(f"[JOB {job_id[:8]}] Frame source: out.frames (len={len(seq) if hasattr(seq,'__len__') else 'na'})")
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
    """PIL / np / tensor 혼재를 [np.uint8 HWC] 리스트로 표준화."""
    frames_u8: List[np.ndarray] = []
    if (hasattr(seq, "cpu") and not isinstance(seq, (list, tuple))) or isinstance(seq, np.ndarray):
        arr = seq.detach().cpu().numpy() if hasattr(seq, "cpu") else seq
        if arr.ndim == 5:
            if arr.shape[2] in (1, 3):
                arr = np.transpose(arr[0], (0, 2, 3, 1))
            else:
                arr = arr[0]
            f = arr.shape[0]
            frames_u8 = [_to_uint8_hwc(arr[i]) for i in range(f)]
            log.info(f"[JOB {job_id[:8]}] seq ndarray (5D) → {f} frames.")
            return frames_u8
        if arr.ndim == 4:
            if arr.shape[1] in (1, 3):
                arr = np.transpose(arr, (0, 2, 3, 1))
            f = arr.shape[0]
            frames_u8 = [_to_uint8_hwc(arr[i]) for i in range(f)]
            log.info(f"[JOB {job_id[:8]}] seq ndarray (4D) → {f} frames.")
            return frames_u8
        if arr.ndim == 3:
            frames_u8 = [_to_uint8_hwc(arr)]
            log.info(f"[JOB {job_id[:8]}] seq ndarray (3D) → 1 frame.")
            return frames_u8
    if isinstance(seq, (list, tuple)):
        for idx, fr in enumerate(seq):
            if hasattr(fr, "cpu"):
                fr = fr.detach().cpu().numpy()
            if isinstance(fr, PILImage.Image):
                fr = np.array(fr)
            if not isinstance(fr, np.ndarray):
                raise TypeError(f"Unsupported frame item at {idx}: {type(fr)}")
            frames_u8.append(_to_uint8_hwc(fr))
        log.info(f"[JOB {job_id[:8]}] seq iterable → {len(frames_u8)} frames.")
        return frames_u8
    raise TypeError(f"Unsupported frame sequence type: {type(seq)}")

# ------------------------------
# 비디오 인코딩 (96% → 100%)
# ------------------------------
def _export_video(job_id: str, frames: List[np.ndarray], out_path: str, fps: int):
    log.info(f"[JOB {job_id[:8]}] encoding video -> {out_path}")
    JOBS[job_id]["progress"] = 0.96
    log.info(f"[JOB {job_id[:8]}] denoise steps finished. waiting for decode/post-processing…")
    if not frames:
        raise RuntimeError("No frames to encode.")
    writer = imageio.get_writer(
        out_path, fps=fps, codec="libx264",
        ffmpeg_params=["-crf", "18", "-preset", "medium"]
    )
    try:
        total = len(frames)
        for i, arr in enumerate(frames, 1):
            writer.append_data(arr)
            JOBS[job_id]["progress"] = round(0.96 + 0.04 * (i / total), 4)
            if i % max(1, total // 10) == 0:
                log.info(f"[JOB {job_id[:8]}] encoding {i}/{total} frames ({100*i/total:.1f}%)")
    finally:
        log.info(f"[JOB {job_id[:8]}] finalizing video file…")
        writer.close()
    log.info(f"[JOB {job_id[:8]}] export done: {out_path}")

# ------------------------------
# 에러 코드 유틸
# ------------------------------
def _classify_error(exc: Exception) -> str:
    msg = str(exc) or ""
    mlow = msg.lower()
    if isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in mlow:
        return "OUT_OF_MEMORY"
    if "hf_transfer" in mlow:
        return "HF_TRANSFER_ERROR"
    if "does not appear to have a file named" in msg or "file not found" in mlow:
        return "MODEL_FILES_MISSING"
    if "divisible by 4" in mlow or "num_frames" in mlow:
        return "INVALID_ARGUMENT"
    if "unexpected keyword argument 'callback'" in mlow:
        return "UNSUPPORTED_CALLBACK_PARAM"
    if "unexpected keyword argument 'callback_steps'" in mlow:
        return "UNSUPPORTED_CALLBACK_STEPS"
    if "frame extraction failed" in mlow:
        return "FRAME_EXTRACTION_ERROR"
    if "truncated" in mlow or "cannot identify image file" in mlow or "image file is truncated" in mlow:
        return "INVALID_IMAGE"
    return "RUNTIME_ERROR"


def _fail_job(job_id: str, exc: Exception):
    code = _classify_error(exc)
    JOBS[job_id]["status"] = "error"
    JOBS[job_id]["error"] = str(exc)
    JOBS[job_id]["error_code"] = code
    log.exception(f"[JOB {job_id[:8]}] ERROR [{code}]: {exc}")

# ------------------------------
# 잡 시작 (main.py에서 호출)
# ------------------------------
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
    download_dir: str = "downloads",
) -> str:
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

            # 총 프레임(정확히) 계산
            total_frames_req = max(1, int(round(fps * duration_sec)))

            # 기본 해상도/프레임 규칙 보정 + 스텝 고정(28)
            steps_local = num_inference_steps
            w0, h0, frames_ruled, steps_fixed = _autoscale_for_vram(width, height, total_frames_req, steps_local, vram_gb=None)
            h, w = _snap_hw(w0, h0, pipe)
            steps_local = steps_fixed  # 28 고정

            # 세그먼트 플랜
            seg_plans = _plan_segments(frames_ruled, SEG_LEN_PREF, OVERLAP)
            num_segments = len(seg_plans)
            total_denoise_steps = max(1, steps_local) * max(1, num_segments)

            plan_mp = _budget_megapixels(w, h, min(MAX_FRAMES_PER_CALL, SEG_LEN_PREF))
            log.info(
                f"[JOB {job_id[:8]}] mode={'I2V' if is_i2v else 'T2V'} | "
                f"req={width}x{height}@{total_frames_req}f,{fps}fps, steps={num_inference_steps} → "
                f"plan: {num_segments} segments (len~{SEG_LEN_PREF}, ov={OVERLAP}) @ {w}x{h}, "
                f"per-call steps={steps_local} (~{plan_mp:.1f} MP/call) | "
                f"gs={guidance_scale} | device={_device} ({_gpu_str()})"
            )

            JOBS[job_id]["progress"] = 0.05

            # 입력 이미지 준비(I2V)
            original_img = _pil_from_bytes(image_bytes) if is_i2v else None
            current_img = None
            if original_img is not None:
                original_img = original_img.resize((w, h), RESAMPLE_BICUBIC)
                current_img = original_img.copy()

            # 결과 프레임 버퍼
            all_frames: List[np.ndarray] = []
            denoise_step_counter = 0  # 전체 진행률 계산용

            # 공용 콜백(세그먼트 인덱스별 진행률 반영)
            def make_on_step_end(seg_index: int):
                def cb_on_step_end(pipe_, step: int, timestep, callback_kwargs):
                    nonlocal denoise_step_counter
                    # 현재 세그 내 step은 0..steps_local-1
                    local_progress = (step + 1) / float(max(1, steps_local))
                    # 세그 시작 전까지 완료된 스텝 수 + 현재 세그의 완료분
                    done_steps = seg_index * max(1, steps_local) + (step + 1)
                    denoise_step_counter = done_steps
                    # 디노이즈 95%까지 할당
                    denoise_pct = min(0.95, denoise_step_counter / float(total_denoise_steps))
                    JOBS[job_id]["progress"] = round(max(JOBS[job_id]["progress"], denoise_pct), 4)
                    if step % max(1, steps_local // 4) == 0:
                        eta_steps = total_denoise_steps - done_steps
                        log.info(
                            f"[JOB {job_id[:8]}] seg {seg_index+1}/{num_segments} "
                            f"step {step+1}/{steps_local} ({local_progress*100:.1f}%) | "
                            f"GLOBAL {denoise_pct*100:.1f}% | ETA ~{eta_steps * 0.0 + 0:.1f}s | GPU {_gpu_str()}"
                        )
                    return callback_kwargs
                return cb_on_step_end

            # ===== 세그먼트 루프 =====
            for si, seg_len in enumerate(seg_plans):
                # I2V: 세그 시작 시 앵커 리셋/블렌드
                if is_i2v and original_img is not None:
                    if USE_ANCHOR_EVERY_SEG:
                        current_img = original_img.copy()
                    elif USE_BLEND and current_img is not None:
                        # 직전 프레임 기반 current_img와 원본 블렌드
                        try:
                            current_img = PILImage.blend(current_img, original_img, ALPHA_ORIG)
                        except Exception:
                            current_img = original_img.copy()
                    else:
                        # 기본: 직전 세그 마지막 프레임을 그대로 사용
                        if current_img is None:
                            current_img = original_img.copy()

                # 파이프 호출
                with torch.inference_mode():
                    ctx = torch.autocast("cuda", dtype=_dtype) if _has_cuda else nullcontext()
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
                        # (T2V 레거시 콜백은 비활성화: 세그 기반 전역 진행률이 더 정확)
                        try:
                            pipe.set_progress_bar_config(disable=True)
                        except Exception:
                            pass
                        out = pipe(**call_kwargs)

                # 프레임 추출
                seg_frames = _extract_frames(job_id, out)  # 길이 seg_len 예상
                if not seg_frames:
                    raise RuntimeError(f"Segment {si} returned no frames.")
                # I2V 다음 세그 준비용 current_img 업데이트: 이번 세그 마지막 프레임
                if is_i2v:
                    try:
                        current_img = PILImage.fromarray(seg_frames[-1])
                    except Exception:
                        current_img = original_img.copy()

                # 합치기(오버랩 크로스페이드)
                if si == 0:
                    all_frames.extend(seg_frames)
                else:
                    # 이전 꼬리와 현재 머리의 overlap 만큼 블렌딩
                    tail = all_frames[-OVERLAP:] if len(all_frames) >= OVERLAP else all_frames[-1:]
                    head = seg_frames[:OVERLAP] if len(seg_frames) >= OVERLAP else seg_frames[:1]
                    blended = _crossfade_frames(tail, head, OVERLAP)
                    # 이전 꼬리 제거(겹칠 부분)
                    if OVERLAP > 0 and len(all_frames) >= len(tail):
                        all_frames = all_frames[:-len(tail)]
                    # 블렌드 + 이번 세그 전체 붙이되, 머리 overlap 부분은 블렌드로 대체했으니 제외
                    all_frames.extend(blended)
                    all_frames.extend(seg_frames[OVERLAP:])

            # 총 길이를 정확히 자르거나 패딩(이론상 정확히 맞음. 안전 보정)
            if len(all_frames) > total_frames_req:
                all_frames = all_frames[:total_frames_req]
            elif len(all_frames) < total_frames_req:
                # 마지막 프레임 반복으로 패딩
                if all_frames:
                    last = all_frames[-1]
                    all_frames.extend([last.copy() for _ in range(total_frames_req - len(all_frames))])

            log.info(f"[JOB {job_id[:8]}] Denoising complete. Collected {len(all_frames)} frames (target {total_frames_req}).")

            # ====== 인코딩 (96% → 100%) ======
            try:
                os.makedirs(download_dir, exist_ok=True)
                out_path = os.path.join(download_dir, f"wan2_{job_id}.mp4")
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
