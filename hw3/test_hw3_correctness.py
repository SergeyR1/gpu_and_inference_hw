"""Self-check tests for HW3 student implementations.

Run from hw3 dir:
    pytest test_hw3_correctness.py
"""

from __future__ import annotations
from engine_utils import Request, BatchPhase
from hw3_task import CacheManager as StudentCacheManager
from hw3_task import Scheduler as StudentScheduler


def test_allocate_can_evict_overlapping_prefix_entries() -> None:
    cm = StudentCacheManager(num_blocks=4, block_size=4)

    req_blocks = cm.allocate(2)
    assert req_blocks is not None

    cm.insert_prefix(tokens=[10, 11, 12, 13, 14, 15, 16, 17], block_ids=req_blocks)
    cm.free(req_blocks)  # keep blocks cached-only (evictable)

    assert cm.num_free_blocks == 2

    # Needs one more block than free pool currently has, so _evict_blocks_from_kv_cache must run.
    claimed = cm.allocate(3)
    assert claimed is not None, "allocate(3) should succeed by evicting cache entries"
    assert len(claimed) == 3


def test_lock_prevents_eviction_until_unlock() -> None:
    cm = StudentCacheManager(num_blocks=2, block_size=4)

    req_blocks = cm.allocate(2)
    assert req_blocks is not None

    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    cm.insert_prefix(tokens=tokens, block_ids=req_blocks)
    cm.free(req_blocks)

    handle = cm.match_prefix(tokens + [999])
    assert handle.num_matched_tokens == 8
    assert len(handle.matched_blocks) == 2

    cm.lock(handle)
    assert cm.allocate(1) is None, "locked cached blocks must not be evicted"

    cm.unlock(handle)
    assert cm.allocate(1) is not None, (
        "after unlock, eviction should make allocation possible"
    )


def test_prefill_admission_uses_prefix_cache() -> None:
    cm = StudentCacheManager(num_blocks=8, block_size=4)

    cached_blocks = cm.allocate(2)
    assert cached_blocks is not None

    shared_prefix = [20, 21, 22, 23, 24, 25, 26, 27]
    cm.insert_prefix(shared_prefix, cached_blocks)
    cm.free(cached_blocks)

    sched = StudentScheduler(
        cache_manager=cm,
        block_size=4,
        max_seqs=2,
        token_budget=8,
        prefill_chunk=8,
        enable_prefix_caching=True,
    )

    req = Request(
        request_id=1,
        arrival_step=0,
        prompt_tokens=shared_prefix + [101, 102],
        max_new_tokens=4,
    )
    sched.add(req)
    batch = sched.schedule()

    assert batch is not None
    assert batch.phase == BatchPhase.PREFILL
    assert batch.is_prefill
    assert req in batch.newly_admitted
    assert req.prefix_tokens_saved == 8
    assert req.num_computed_tokens == 8
    assert req.cache_handle is not None
    assert len(req.block_table) >= 2


def test_decode_allocates_on_block_boundary() -> None:
    cm = StudentCacheManager(num_blocks=4, block_size=4)
    sched = StudentScheduler(
        cache_manager=cm,
        block_size=4,
        max_seqs=1,
        token_budget=8,
        prefill_chunk=8,
        enable_prefix_caching=False,
    )

    req = Request(
        request_id=7,
        arrival_step=0,
        prompt_tokens=[1, 2, 3, 4],
        max_new_tokens=3,
    )
    initial = cm.allocate(1)
    assert initial is not None

    req.block_table = list(initial)
    req.num_computed_tokens = 4
    req.status = "RUNNING"
    sched.running.append(req)

    batch = sched.schedule()
    assert batch is not None
    assert batch.phase == BatchPhase.DECODE
    assert batch.is_decode
    assert req in batch.to_decode
    assert len(req.block_table) == 2, (
        "decode token at position 4 should use a new block"
    )
