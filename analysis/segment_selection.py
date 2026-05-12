from __future__ import annotations

from typing import Iterable, List

from analysis.types import SegmentAttribution


def _overlap_ratio(a: SegmentAttribution, b: SegmentAttribution) -> float:
    inter = max(0.0, min(a.end_sec, b.end_sec) - max(a.start_sec, b.start_sec))
    if inter <= 0.0:
        return 0.0
    shorter = max(1e-8, min(a.end_sec - a.start_sec, b.end_sec - b.start_sec))
    return inter / shorter


def assign_overlap_groups(segments: Iterable[SegmentAttribution], overlap_threshold: float) -> List[SegmentAttribution]:
    ordered = sorted(segments, key=lambda seg: (seg.start_sec, seg.end_sec))
    group_id = 0
    anchors: list[SegmentAttribution] = []
    for seg in ordered:
        assigned = False
        for anchor in anchors:
            if _overlap_ratio(seg, anchor) >= overlap_threshold:
                seg.overlap_group_id = anchor.overlap_group_id
                assigned = True
                break
        if not assigned:
            seg.overlap_group_id = group_id
            anchors.append(seg)
            group_id += 1
    return ordered


def _sort_key(seg: SegmentAttribution):
    if seg.rank_type == "exact_match":
        return (-seg.target_probability, -seg.combined_score, seg.start_sec)
    return (-seg.combined_score, -seg.target_probability, seg.start_sec)


def select_diverse_segments(
    segments: Iterable[SegmentAttribution],
    top_n: int,
    overlap_threshold: float,
    existing: Iterable[SegmentAttribution] | None = None,
) -> List[SegmentAttribution]:
    chosen: List[SegmentAttribution] = list(existing or [])
    base_count = len(chosen)
    for seg in sorted(segments, key=_sort_key):
        if any(_overlap_ratio(seg, kept) >= overlap_threshold for kept in chosen):
            continue
        chosen.append(seg)
        if len(chosen) - base_count >= top_n:
            break
    return chosen[base_count:]
