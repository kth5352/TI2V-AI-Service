# backend/worker.py
"""
잡 오케스트레이션:
- start_job(): 잡 생성/스레드 시작
- 세그먼트 루프 수행, 진행률 갱신, 프레임 합성, 인코딩
- 나머지 세부 로직은 core/utils 모듈에 위임
"""

from __future__ import annotations

import os
import time
import uuid
import threading
from typing import Optional, List

import numpy as np
from PIL import Image as PILImage

from core.config import (
    MEDIA_DIR, SEG_LEN_PREF, OVERLAP, MAX_FRAMES_PER_CALL,
    USE_ANCHOR_EVERY_SEG, USE_BLEND, ALPHA_ORIG,
    DEVICE, DTYPE, HAS_CUDA,
)
from core.jobs import JOBS
from core.pipelines import get_i2v_pipe, get_t2v_pipe, snap_hw, autoscale_for_vram, gpu_str
from core.segments import plan_segments, crossfade_frames
from core.errors import fail_job
from utils.image_io import pil_from_bytes, RESAMPLE_BICUBIC
from utils.frames import extract_frames
from utils.encode import export_video


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
    - 성공: status=done, result_path 설정
    - 실패: status=error, error/error_code 설정
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
        import torch  # 지연 임포트(서버 시작 속도/메모리 이점)
        from contextlib import nullcontext
        import inspect
        import logging
        log = logging.getLogger("wan-i2v")

        t0 = time.time()
        try:
            JOBS[job_id]["status"] = "running"
            is_i2v = image_bytes is not None
            pipe = get_i2v_pipe() if is_i2v else get_t2v_pipe()

            # 총 프레임 정확 계산
            total_frames_req = max(1, int(round(fps * duration_sec)))

            # 규약 보정(프레임) + 스텝 28 고정
            steps_local = num_inference_steps
            w0, h0, frames_ruled, steps_fixed = autoscale_for_vram(width, height, total_frames_req, steps_local, vram_gb=None)
            h, w = snap_hw(w0, h0, pipe)  # (H,W)
            steps_local = steps_fixed

            # 세그먼트 플랜
            seg_plans = plan_segments(frames_ruled, SEG_LEN_PREF, OVERLAP)
            num_segments = len(seg_plans)
            total_denoise_steps = max(1, steps_local) * max(1, num_segments)

            plan_mp = (w * h * min(MAX_FRAMES_PER_CALL, SEG_LEN_PREF)) / 1_000_000.0
            log.info(
                f"[JOB {job_id[:8]}] mode={'I2V' if is_i2v else 'T2V'} | "
                f"req={width}x{height}@{total_frames_req}f,{fps}fps, steps={num_inference_steps} → "
                f"plan: {num_segments} segments (~{SEG_LEN_PREF}f, ov={OVERLAP}) @ {w}x{h}, "
                f"per-call steps={steps_local} (~{plan_mp:.1f} MP/call) | "
                f"gs={guidance_scale} | device={DEVICE} ({gpu_str()})"
            )

            JOBS[job_id]["progress"] = 0.05

            # 입력 이미지(I2V)
            original_img = pil_from_bytes(image_bytes) if is_i2v else None
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
                # I2V: 세그 시작 시 앵커 리셋/블렌드
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
                    ctx = torch.autocast("cuda", dtype=DTYPE) if HAS_CUDA else nullcontext()
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
                seg_frames = extract_frames(job_id, out)
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
                    blended = crossfade_frames(tail, head, OVERLAP)
                    if OVERLAP > 0 and len(all_frames) >= len(tail):
                        all_frames = all_frames[:-len(tail)]
                    all_frames.extend(blended)
                    all_frames.extend(seg_frames[OVERLAP:])

            # 총 길이 보정(안전)
            if len(all_frames) > total_frames_req:
                all_frames = all_frames[:total_frames_req]
            elif len(all_frames) < total_frames_req and all_frames:
                last = all_frames[-1]
                all_frames.extend([last.copy() for _ in range(total_frames_req - len(all_frames))])

            # ===== 인코딩 =====
            out_path = os.path.join(download_dir, f"wan2_{job_id}.mp4")
            try:
                export_video(job_id, all_frames, out_path, fps=fps)
            except Exception as e:
                fail_job(job_id, e)
                return

            JOBS[job_id].update(status="done", progress=1.0, result_path=out_path)

        except Exception as e:
            fail_job(job_id, e)
        finally:
            # 끝 로그
            import logging
            logging.getLogger("wan-i2v").info(f"[JOB {job_id[:8]}] CLOSED in {time.time()-t0:.1f}s")

    threading.Thread(target=run, daemon=True).start()
    return job_id


# main.py 호환성 유지를 위한 re-export
MEDIA_DIR = MEDIA_DIR  # 그대로 노출
