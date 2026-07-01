"""GPU selection: parse nvidia-smi output and pick a free card (no GPU needed)."""

from grace.train import GPU_FREE_MEM_MB, _parse_gpu_stats, pick_free_gpu

SAMPLE = """0, 40000, 46068, 95
1, 12, 46068, 0
2, 300, 46068, 3
"""


def test_parse_gpu_stats():
    stats = _parse_gpu_stats(SAMPLE)
    assert [g["index"] for g in stats] == [0, 1, 2]
    assert stats[0]["mem_used"] == 40000 and stats[0]["util"] == 95
    assert stats[1]["mem_total"] == 46068


def test_pick_free_gpu_prefers_least_used_idle_card():
    stats = _parse_gpu_stats(SAMPLE)
    # gpu0 is busy (40 GB); gpu1 (12 MB) and gpu2 (300 MB) are idle -> pick gpu1
    assert pick_free_gpu(stats) == 1


def test_pick_free_gpu_returns_none_when_all_busy():
    busy = [{"index": i, "mem_used": GPU_FREE_MEM_MB + 1000, "mem_total": 46068, "util": 50} for i in range(4)]
    assert pick_free_gpu(busy) is None
