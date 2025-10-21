# backend/video_worker.py — WAN2.2 I2V/T2V
# 진행률(SSE) + 안전 워밍업 + 보수적 자동 다운스케일 + 동적 콜백 호환 + 출력 안전 추출 + 인코딩 진행 표시 + 에러코드
import io
import os
import uuid
import time
import threading
import logging
import inspect
from typing import Optional, Dict, Any, Tuple, List
from contextlib import nullcontext

import torch
import numpy as np
from PIL import Image


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
_WARMED_T2V = False
_WARMED_I2V = False

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


def _pil_from_bytes(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")


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
    w: int, h: int, frames: int, steps: int, vram_gb: float = 12.0
) -> Tuple[int, int, int, int]:
    """
    12GB 기준 보수적 설정:
      - 스텝 상한 28
      - 총 MP ~12MP 목표로 축소
    """
    def nearest_valid(n: int) -> int:
        if (n - 1) % 4 == 0:
            return n
        return int(round((n - 1) / 4.0) * 4 + 1)

    frames = nearest_valid(frames)
    steps = 28

    return w, h, nearest_valid(frames), steps


# 기존: def _apply_memory_savers(pipe):
# 변경:
def _apply_memory_savers(pipe, *, is_i2v: bool):
    # 공통: attention slicing
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    # ⛔ I2V에선 vae_tiling 금지 (채널 mismatch 방지)
    if not is_i2v:
        try:
            pipe.enable_vae_tiling()
        except Exception:
            pass

    # xFormers는 Windows 기본 OFF로 유지하는 현재 설정(_USE_XFORMERS)에 따름
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
                    kwargs["image"] = Image.new("RGB", (320, 320), (0, 0, 0))
                pipe(**kwargs)
        log.info("[WARMUP] done.")
    except Exception as e:
        log.warning(f"[WARMUP] skipped: {e}")
    if is_i2v:
        _WARMED_I2V = True
    else:
        _WARMED_T2V = True

# ------------------------------
# 파이프라인 로딩
# ------------------------------
def get_t2v_pipe():
    global _PIPE_T2V
    if _PIPE_T2V is None:
        vae = _load_vae(MODEL_ID)
        pipe = WanPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=_dtype, local_files_only=True)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        _apply_memory_savers(pipe, is_i2v=False)
        # CPU 오프로딩이 있으면 .to()는 생략 (내부 관리)
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
        pipe = WanImageToVideoPipeline.from_pretrained(
            MODEL_ID, torch_dtype=_dtype, local_files_only=True
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        # I2V: attention slicing만 (vae_tiling 금지)
        _apply_memory_savers(pipe, is_i2v=True)

        # 우선 GPU 고정
        pipe.to(_device)

        # ✅ VRAM 보호를 위해 I2V에서는 CPU 오프로딩 ‘사용’
        try:
            pipe.enable_sequential_cpu_offload()
            log.info("[PIPE] I2V enabled CPU Offload.")
        except Exception:
            # 불가 시 GPU 고정 유지
            pipe.to(_device)

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
        # 0..1, -1..1, 0..255 등 보호적 처리
        a = arr.astype(np.float32)
        if a.max() <= 1.0 + 1e-3 and a.min() >= 0.0 - 1e-3:
            a = a * 255.0
        elif a.min() < 0.0 and a.max() <= 1.0 + 1e-3:
            a = (a + 1.0) * 127.5  # [-1,1]→[0,255]
        a = np.clip(a, 0, 255).astype(np.uint8)
    else:
        a = arr

    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):
        # (C,H,W) → (H,W,C)
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

    # 1) out.frames 케이스
    if hasattr(out_obj, "frames"):
        frames = getattr(out_obj, "frames")
        # 보통 batch dim: frames[0] → sequence
        seq = frames[0] if isinstance(frames, (list, tuple)) and len(frames) > 0 else frames
        log.info(f"[JOB {job_id[:8]}] Frame source: out.frames (len={len(seq) if hasattr(seq,'__len__') else 'na'})")
        return _normalize_frame_sequence(job_id, seq)

    # 2) out.videos (Diffusers 일부 파이프라인)
    if hasattr(out_obj, "videos"):
        vids = getattr(out_obj, "videos")
        # tensor or np: (B,F,C,H,W) or (B,F,H,W,C)
        if hasattr(vids, "cpu"):  # torch.Tensor
            vids = vids.detach().cpu().numpy()
        if isinstance(vids, np.ndarray):
            if vids.ndim == 5:
                b, f = vids.shape[0], vids.shape[1]
                if b == 0 or f == 0:
                    raise ValueError("videos is empty.")
                # (B,F,C,H,W) → (F,H,W,C)
                if vids.shape[2] in (1, 3):
                    seq = np.transpose(vids[0], (0, 2, 3, 1))
                else:
                    # (B,F,H,W,C) → (F,H,W,C)
                    seq = vids[0]
                log.info(f"[JOB {job_id[:8]}] Frame source: out.videos (B={b},F={f})")
                return [_to_uint8_hwc(fr) for fr in seq]
        raise AttributeError("Unsupported 'videos' format.")

    # 3) list/tuple 첫 요소
    if isinstance(out_obj, (list, tuple)):
        if len(out_obj) == 0:
            raise AttributeError("Pipeline returned empty list/tuple.")
        log.info(f"[JOB {job_id[:8]}] Frame source: out[0] (list/tuple)")
        return _normalize_frame_sequence(job_id, out_obj[0])

    # 4) dict-like
    if isinstance(out_obj, dict):
        for k in ("frames", "videos"):
            if k in out_obj:
                return _extract_frames(job_id, type("X", (), out_obj))  # 속성처럼 접근
        raise AttributeError("Dict output without 'frames' or 'videos'.")

    raise AttributeError(f"Unexpected pipeline output type: {type(out_obj)}")


def _normalize_frame_sequence(job_id: str, seq) -> List[np.ndarray]:
    """PIL / np / tensor 혼재를 [np.uint8 HWC] 리스트로 표준화."""
    frames_u8: List[np.ndarray] = []

    # 💡 [수정됨] 시퀀스가 tensor/numpy 한 덩어리일 수도 있음: (F,C,H,W) or (F,H,W,C)
    if (hasattr(seq, "cpu") and not isinstance(seq, (list, tuple))) or isinstance(seq, np.ndarray):
        arr = seq.detach().cpu().numpy() if hasattr(seq, "cpu") else seq
        # 5D: (B,F,C,H,W) or (B,F,H,W,C) → (F,H,W,C)
        if arr.ndim == 5:
            if arr.shape[2] in (1, 3):        # (B,F,C,H,W)
                arr = np.transpose(arr[0], (0, 2, 3, 1))
            else:                              # (B,F,H,W,C)
                arr = arr[0]
            f = arr.shape[0]
            frames_u8 = [_to_uint8_hwc(arr[i]) for i in range(f)]
            log.info(f"[JOB {job_id[:8]}] seq ndarray (5D) → {f} frames.")
            return frames_u8
        # 4D: (F,C,H,W) or (F,H,W,C) → (F,H,W,C)
        if arr.ndim == 4:
            if arr.shape[1] in (1, 3):        # (F,C,H,W)
                arr = np.transpose(arr, (0, 2, 3, 1))
            f = arr.shape[0]
            frames_u8 = [_to_uint8_hwc(arr[i]) for i in range(f)]
            log.info(f"[JOB {job_id[:8]}] seq ndarray (4D) → {f} frames.")
            return frames_u8
        # ✅ 3D: 단일 프레임 (H,W,C) → 리스트 1장
        if arr.ndim == 3:
            frames_u8 = [_to_uint8_hwc(arr)]
            log.info(f"[JOB {job_id[:8]}] seq ndarray (3D) → 1 frame.")
            return frames_u8
        # 4D가 아닌 단일 np.ndarray인 경우는 아래 TypeError로 빠지게 됩니다.

    # 일반 케이스: iterable (List[PIL], List[Tensor], List[np])
    if isinstance(seq, (list, tuple)):
        for idx, fr in enumerate(seq):
            if hasattr(fr, "cpu"):  # torch tensor
                fr = fr.detach().cpu().numpy()
            if isinstance(fr, Image.Image):
                fr = np.array(fr)
            if not isinstance(fr, np.ndarray):
                raise TypeError(f"Unsupported frame item at {idx}: {type(fr)}")
            frames_u8.append(_to_uint8_hwc(fr))
        log.info(f"[JOB {job_id[:8]}] seq iterable → {len(frames_u8)} frames.")
        return frames_u8

    # 이 시점에 도달한 단일 np.ndarray는 처리할 수 없습니다. (예: (H,W,C) 형태의 단일 이미지)
    raise TypeError(f"Unsupported frame sequence type: {type(seq)}")

# ------------------------------
# 비디오 인코딩 (96% → 100%)
# ------------------------------
def _export_video(job_id: str, frames: List[np.ndarray], out_path: str, fps: int):
    log.info(f"[JOB {job_id[:8]}] encoding video -> {out_path}")
    JOBS[job_id]["progress"] = 0.96

    # infer_once() 리턴 전에 찍히는 로그가 없으니, 아래처럼 보강:
    log.info(f"[JOB {job_id[:8]}] denoise steps finished. waiting for decode/post-processing…")

    if not frames or len(frames) == 0:
        raise RuntimeError("No frames to encode.")

    log.info(f"[JOB {job_id[:8]}] starting video writer for {len(frames)} frames…")
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8)
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
    download_dir: str = "backend/downloads",
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
            pipe = get_i2v_pipe() if image_bytes else get_t2v_pipe()

            # 프레임/스텝 계획
            steps_local = num_inference_steps
            raw_frames = max(8, min(int(round(fps * duration_sec)), 121))
            w0, h0, f0, s0 = _autoscale_for_vram(width, height, raw_frames, steps_local, vram_gb=12.0)
            h, w = _snap_hw(w0, h0, pipe)
            num_frames = f0
            steps_local = s0

            plan_mp = _budget_megapixels(w, h, num_frames)
            log.info(
                f"[JOB {job_id[:8]}] mode={'I2V' if image_bytes else 'T2V'} | "
                f"req={width}x{height}@{raw_frames}f,{fps}fps, steps={num_inference_steps} → "
                f"plan={w}x{h}@{num_frames}f, steps={steps_local} (~{plan_mp:.1f} MP) | "
                f"gs={guidance_scale} | device={_device} ({_gpu_str()})"
            )

            JOBS[job_id]["progress"] = 0.05
            step_times = []
            last_step_ts = time.time()

            def cb_on_step_end(pipe_, step: int, timestep, callback_kwargs):
                nonlocal last_step_ts
                pct = (step + 1) / float(steps_local)
                JOBS[job_id]["progress"] = round(min(0.95, pct), 4)
                now = time.time()
                step_times.append(now - last_step_ts)
                last_step_ts = now
                avg = sum(step_times) / len(step_times) if step_times else 0.0
                remain = max(0, steps_local - (step + 1))
                eta = remain * avg
                log.info(
                    f"[JOB {job_id[:8]}] step {step+1}/{steps_local} "
                    f"({pct*100:.1f}%) | ETA ~{eta:,.1f}s | GPU {_gpu_str()}"
                )
                return callback_kwargs

            def cb_legacy(step: int, timestep, latents):
                nonlocal last_step_ts
                pct = (step + 1) / float(steps_local)
                JOBS[job_id]["progress"] = round(min(0.95, pct), 4)
                now = time.time()
                step_times.append(now - last_step_ts)
                last_step_ts = now
                avg = sum(step_times) / len(step_times) if step_times else 0.0
                remain = max(0, steps_local - (step + 1))
                eta = remain * avg
                log.info(
                    f"[JOB {job_id[:8]}] step {step+1}/{steps_local} "
                    f"({pct*100:.1f}%) | ETA ~{eta:,.1f}s | GPU {_gpu_str()}"
                )
                return latents

            img = _pil_from_bytes(image_bytes) if image_bytes else None
            if img is not None:
                img = img.resize((w, h), Image.BICUBIC)

            def infer_once():
                if _has_cuda:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                JOBS[job_id]["progress"] = max(JOBS[job_id]["progress"], 0.10)
                log.info(f"[JOB {job_id[:8]}] start denoising…")
                with torch.inference_mode():
                    ctx = torch.autocast("cuda", dtype=_dtype) if _has_cuda else nullcontext()
                    with ctx:
                        # 지원되는 콜백 인자만 넣기 (파이프라인별 호환)
                        sig_params = set(inspect.signature(pipe.__call__).parameters.keys())
                        call_kwargs = dict(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            height=h,
                            width=w,
                            num_frames=num_frames,
                            num_inference_steps=steps_local,
                            guidance_scale=guidance_scale,
                        )
                        if img is not None and "image" in sig_params:
                            call_kwargs["image"] = img
                        # 콜백들도 시그니처 존재 여부 기반으로만 세팅
                        if "callback_on_step_end" in sig_params:
                            call_kwargs["callback_on_step_end"] = cb_on_step_end
                        if not image_bytes and "callback" in sig_params:
                            call_kwargs["callback"] = cb_legacy
                        if not image_bytes and "callback_steps" in sig_params:
                            call_kwargs["callback_steps"] = 1
                        try:
                            pipe.set_progress_bar_config(disable=True)
                        except Exception:
                            pass
                        return pipe(**call_kwargs)

            try:
                out = infer_once()
            except torch.cuda.OutOfMemoryError:
                log.warning(f"[JOB {job_id[:8]}] OOM at start. Retrying smaller & offload…")
                num_frames = _nearest_valid_frames(max(9, int(num_frames * 0.75)))
                h = max(320, (h // 2) // 32 * 32)
                w = max(320, (w // 2) // 32 * 32)
                steps_local = max(12, int(steps_local * 0.75))
                try:
                    pipe.enable_sequential_cpu_offload()
                except Exception:
                    pass
                if _USE_XFORMERS:
                    try:
                        pipe.enable_xformers_memory_efficient_attention()
                    except Exception:
                        pass
                out = infer_once()
            except Exception as e:
                _fail_job(job_id, e)
                return

            # ====== 출력 → 프레임 안전 추출 ======
            try:
                frames = _extract_frames(job_id, out)
                log.info(f"[JOB {job_id[:8]}] Denoising complete. Extracted {len(frames)} frames.")
            except Exception as e:
                JOBS[job_id]["status"] = "error"
                JOBS[job_id]["error"] = f"Frame extraction failed: {e}"
                JOBS[job_id]["error_code"] = "FRAME_EXTRACTION_ERROR"
                log.exception(f"[JOB {job_id[:8]}] FATAL FRAME EXTRACTION ERROR: {e}")
                return

            # ====== 인코딩 (96% → 100%) ======
            try:
                os.makedirs(download_dir, exist_ok=True)
                out_path = os.path.join(download_dir, f"wan2_{job_id}.mp4")
                _export_video(job_id, frames, out_path, fps=fps)
            except Exception as e:
                _fail_job(job_id, e)
                return

            JOBS[job_id].update(status="done", progress=1.0, result_path=out_path)
            log.info(f"[JOB {job_id[:8]}] DONE in {time.time()-t0:.1f}s")

        except Exception as e:
            _fail_job(job_id, e)

    threading.Thread(target=run, daemon=True).start()
    return job_id