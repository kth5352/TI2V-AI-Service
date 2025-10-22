# backend/core/pipelines.py
"""
Diffusers 파이프라인 로딩/워밍업/옵션 적용
"""
from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Tuple

import torch
from diffusers import WanPipeline, WanImageToVideoPipeline, UniPCMultistepScheduler
try:
    from diffusers import AutoencoderKLWan
except Exception:
    AutoencoderKLWan = None

from core.config import MODEL_ID, DEVICE, DTYPE, HAS_CUDA, USE_XFORMERS

log = logging.getLogger("wan-i2v")

_PIPE_T2V = None
_PIPE_I2V = None
_WARMED_T2V = False
_WARMED_I2V = False
from threading import Lock
_PIPE_LOCK = Lock()


def gpu_str() -> str:
    """GPU/메모리 상태 문자열(로그용)"""
    if not HAS_CUDA:
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
    return f"{name} | {mem}"


def _load_vae(model_id: str):
    if AutoencoderKLWan is not None:
        return AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    return None


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
    if USE_XFORMERS:
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass


def _warmup_once(pipe, is_i2v: bool):
    global _WARMED_T2V, _WARMED_I2V
    if (is_i2v and _WARMED_I2V) or ((not is_i2v) and _WARMED_T2V):
        return
    try:
        log.info("[WARMUP] tiny warmup…")
        with torch.inference_mode():
            ctx = torch.autocast("cuda", dtype=DTYPE) if HAS_CUDA else nullcontext()
            with ctx:
                try:
                    pipe.set_progress_bar_config(disable=True)
                except Exception:
                    pass
                kwargs = dict(prompt="warmup", height=256, width=256, num_frames=5, num_inference_steps=1, guidance_scale=2.5)
                if is_i2v:
                    from PIL import Image as PILImage
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
                pipe = WanPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=DTYPE, local_files_only=True)
                pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
                _apply_memory_savers(pipe, is_i2v=False)
                try:
                    pipe.enable_sequential_cpu_offload()
                    log.info("[PIPE] T2V enabled CPU Offload.")
                except Exception:
                    pipe.to(DEVICE)
                _PIPE_T2V = pipe
                log.info(f"[PIPE] T2V loaded on {DEVICE} ({gpu_str()}); dtype={DTYPE}")
                _warmup_once(_PIPE_T2V, is_i2v=False)
    return _PIPE_T2V


def get_i2v_pipe():
    global _PIPE_I2V
    if _PIPE_I2V is None:
        with _PIPE_LOCK:
            if _PIPE_I2V is None:
                pipe = WanImageToVideoPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE, local_files_only=True)
                pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
                _apply_memory_savers(pipe, is_i2v=True)
                try:
                    pipe.enable_model_cpu_offload()
                    log.info("[PIPE] I2V enabled MODEL CPU Offload.")
                except Exception:
                    pipe.to(DEVICE)
                try:
                    pipe.vae.to(DEVICE)
                    log.info("[PIPE] I2V VAE pinned on GPU.")
                except Exception:
                    pass
                log.info(f"[PIPE] I2V loaded on {DEVICE} ({gpu_str()}); dtype={DTYPE}")
                _warmup_once(pipe, is_i2v=True)
                _PIPE_I2V = pipe
    return _PIPE_I2V


def snap_hw(width: int, height: int, pipe) -> Tuple[int, int]:
    """
    모델 패치/스케일 배수에 맞춰 해상도 스냅. 실패 시 32배수로 폴백.
    반환 (H, W)
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


def nearest_valid_frames(n: int) -> int:
    """WAN 규약: (num_frames-1)%4 == 0"""
    if (n - 1) % 4 == 0:
        return n
    r = int(round((n - 1) / 4.0) * 4 + 1)
    return max(9, r)


def autoscale_for_vram(w: int, h: int, frames: int, steps: int, vram_gb: float | None = None):
    """프레임 규칙 보정 + 스텝 28 고정"""
    frames = nearest_valid_frames(frames)
    steps = 28
    return w, h, frames, steps
