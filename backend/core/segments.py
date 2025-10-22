# backend/core/segments.py
"""
세그먼트 플래너와 크로스페이드
"""
from __future__ import annotations
from typing import List
import numpy as np

from core.pipelines import nearest_valid_frames
from core.config import SEG_LEN_PREF, OVERLAP, MAX_FRAMES_PER_CALL


def plan_segments(total_frames: int, seg_len_pref: int = SEG_LEN_PREF, overlap: int = OVERLAP) -> List[int]:
    """
    총 프레임을 세그먼트로 분할.
    - 첫 세그: 오버랩 없음
    - 이후: overlap 길이로 겹치며 (n-1)%4==0 유지
    - 최종 합성 결과가 total_frames가 되도록 미세 조정
    """
    if total_frames <= 0:
        return []
    segs: List[int] = []
    eff_acc = 0  # 오버랩 제외 유효 누적

    pref = nearest_valid_frames(max(9, seg_len_pref))
    while eff_acc < total_frames:
        if not segs:
            need = total_frames - eff_acc
            L = min(pref, need if need <= MAX_FRAMES_PER_CALL else pref)
            L = nearest_valid_frames(max(9, min(MAX_FRAMES_PER_CALL, L)))
            segs.append(L)
            eff_acc += L
        else:
            need = total_frames - eff_acc
            L_eff_target = min(pref - overlap, need)
            L = nearest_valid_frames(max(9, min(MAX_FRAMES_PER_CALL, L_eff_target + overlap)))
            if L - overlap <= 0:
                L = nearest_valid_frames(max(9, overlap + 1))
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
                segs[-1] = nearest_valid_frames(target)
            else:
                target_eff = max(1, (L_last - overlap) - delta)
                segs[-1] = nearest_valid_frames(target_eff + overlap)

    return segs


def crossfade_frames(tail: List[np.ndarray], head: List[np.ndarray], overlap: int) -> List[np.ndarray]:
    """
    tail[-overlap:], head[:overlap]를 선형 가중으로 블렌딩
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
