"""Small, focused correctness tests for Scheduler."""

from __future__ import annotations

from engine_utils import (
    Request,
    RequestStatus,
    SchedulingPolicy,
    BatchPhase,
)
from hw3_task import CacheManager as StudentCacheManager
from hw3_task import Scheduler as StudentScheduler


def _make_scheduler(
    *,
    cache_manager: StudentCacheManager | None = None,
    num_blocks: int = 8,
    block_size: int = 4,
    max_seqs: int = 2,
    token_budget: int = 8,
    prefill_chunk: int = 4,
    enable_prefix_caching: bool = False,
    scheduling_policy: SchedulingPolicy = SchedulingPolicy.PREFILL_FIRST,
) -> tuple[StudentCacheManager, StudentScheduler]:
    cm = cache_manager or StudentCacheManager(
        num_blocks=num_blocks, block_size=block_size
    )
    sched = StudentScheduler(
        cache_manager=cm,
        block_size=block_size,
        max_seqs=max_seqs,
        token_budget=token_budget,
        prefill_chunk=prefill_chunk,
        enable_prefix_caching=enable_prefix_caching,
        scheduling_policy=scheduling_policy,
    )
    return cm, sched


def _allocate_or_fail(cm: StudentCacheManager, n: int) -> list[int]:
    blocks = cm.allocate(n)
    assert blocks is not None
    return list(blocks)


def _attach_running_request(
    *,
    sched: StudentScheduler,
    cm: StudentCacheManager,
    req: Request,
    block_count: int,
    num_computed_tokens: int,
    num_generated_tokens: int = 0,
) -> None:
    req.status = RequestStatus.RUNNING
    req.num_computed_tokens = num_computed_tokens
    req.num_generated_tokens = num_generated_tokens
    req.block_table = _allocate_or_fail(cm, block_count)
    sched.running.append(req)


def test_schedule_returns_none_when_idle_and_advances_step() -> None:
    _, sched = _make_scheduler(
        num_blocks=4, block_size=4, max_seqs=2, token_budget=8, prefill_chunk=4
    )

    assert sched.step == 0
    # Real engine behavior: a step still "ticks" even with no runnable work.
    assert sched.schedule() is None
    assert sched.step == 1


def test_prefill_admits_waiting_request_with_expected_chunk() -> None:
    _, sched = _make_scheduler(
        num_blocks=8, block_size=4, max_seqs=1, token_budget=4, prefill_chunk=4
    )
    req = Request(
        request_id=1,
        arrival_step=0,
        prompt_tokens=[1, 2, 3, 4, 5, 6],
        max_new_tokens=2,
    )
    # New arrivals enter WAITING first, then scheduler admits them.
    sched.add(req)

    batch = sched.schedule()
    assert batch is not None
    assert batch.phase == BatchPhase.PREFILL
    assert batch.is_prefill
    assert batch.newly_admitted == [req]
    # Prefill chunk is capped by both per-step token budget and prefill_chunk.
    assert batch.to_prefill == [(req, 4)]
    assert req.status == RequestStatus.RUNNING
    assert req.first_scheduled_step == 0
    # Admission allocates prompt KV blocks up front.
    assert len(req.block_table) == 2  # ceil(6 / 4)
    assert len(sched.waiting) == 0
    assert req in sched.running


def test_prefill_first_policy_prioritizes_prefill_work() -> None:
    cm, sched = _make_scheduler(
        num_blocks=8,
        block_size=4,
        max_seqs=2,
        token_budget=8,
        prefill_chunk=4,
        scheduling_policy=SchedulingPolicy.PREFILL_FIRST,
    )
    decode_ready = Request(1, 0, [1, 2, 3, 4], 2)
    _attach_running_request(
        sched=sched,
        cm=cm,
        req=decode_ready,
        block_count=1,
        num_computed_tokens=decode_ready.prompt_len,
    )

    waiting = Request(2, 0, [9, 8, 7, 6], 2)
    sched.add(waiting)

    batch = sched.schedule()
    assert batch is not None
    assert batch.is_prefill
    # Prefill-first policy should admit/process waiting prefill before decode.
    assert waiting in batch.newly_admitted


def test_decode_first_policy_prioritizes_decode_work() -> None:
    cm, sched = _make_scheduler(
        num_blocks=8,
        block_size=4,
        max_seqs=2,
        token_budget=8,
        prefill_chunk=4,
        scheduling_policy=SchedulingPolicy.DECODE_FIRST,
    )
    decode_ready = Request(1, 0, [1, 2, 3, 4], 2)
    _attach_running_request(
        sched=sched,
        cm=cm,
        req=decode_ready,
        block_count=1,
        num_computed_tokens=decode_ready.prompt_len,
    )

    waiting = Request(2, 0, [9, 8, 7, 6], 2)
    sched.add(waiting)

    batch = sched.schedule()
    assert batch is not None
    assert batch.phase == BatchPhase.DECODE
    assert batch.is_decode
    assert decode_ready in batch.to_decode
    # Waiting request remains queued this step because decode work is prioritized.
    assert len(sched.waiting) == 1


def test_prefill_admission_uses_prefix_cache_match() -> None:
    cm = StudentCacheManager(num_blocks=8, block_size=4)
    shared_prefix = [20, 21, 22, 23, 24, 25, 26, 27]

    cached_block_ids = _allocate_or_fail(cm, 2)
    # Simulate an earlier completed request that left reusable cached prefix blocks.
    cm.insert_prefix(shared_prefix, cached_block_ids)
    # Two blocks are owned by both the request and cache metadata (2 refs each).
    assert sum(cm.ref_counts) == 4
    # Drop request ownership; cache still owns these blocks for future matches.
    cm.free(cached_block_ids)

    # After free(), only cache ownership remains on those two blocks.
    assert sum(cm.ref_counts) == 2
    # all from [first_part] and [first_part, second_part]
    assert sum(cm.cache_ref_counts) == 3

    _, sched = _make_scheduler(
        cache_manager=cm,
        num_blocks=8,
        block_size=4,
        max_seqs=2,
        token_budget=8,
        prefill_chunk=8,
        enable_prefix_caching=True,
    )
    req = Request(
        request_id=3,
        arrival_step=0,
        prompt_tokens=shared_prefix + [101, 102],
        max_new_tokens=2,
    )
    sched.add(req)

    batch = sched.schedule()
    assert batch is not None
    assert batch.is_prefill
    assert req in batch.newly_admitted
    # Scheduler should skip prefill compute for the matched prefix and keep a handle
    # so those blocks stay pinned for this live request.
    assert req.prefix_tokens_saved == 8
    assert req.num_computed_tokens == 8
    assert req.cache_handle is not None


def test_prefill_first_uses_decode_when_no_prefill_work_exists() -> None:
    cm, sched = _make_scheduler(
        num_blocks=8,
        block_size=4,
        max_seqs=3,
        token_budget=8,
        prefill_chunk=4,
        scheduling_policy=SchedulingPolicy.PREFILL_FIRST,
    )

    decode_ready = Request(10, 0, [1, 2, 3, 4], 3)
    _attach_running_request(
        sched=sched,
        cm=cm,
        req=decode_ready,
        block_count=1,
        num_computed_tokens=decode_ready.prompt_len,
    )

    batch = sched.schedule()
    assert batch is not None
    assert batch.is_decode
    assert batch.to_decode == [decode_ready]


def test_prefill_running_request_is_preempted_when_prompt_blocks_underallocated() -> (
    None
):
    cm, sched = _make_scheduler(
        num_blocks=1, block_size=4, max_seqs=1, token_budget=4, prefill_chunk=4
    )
    req = Request(
        request_id=20,
        arrival_step=0,
        prompt_tokens=[1, 2, 3, 4, 5, 6, 7, 8],  # needs 2 blocks total
        max_new_tokens=2,
    )
    # Under full-admission semantics, running prefill requests must already own
    # full prompt allocation. Violations are preempted.
    _attach_running_request(
        sched=sched,
        cm=cm,
        req=req,
        block_count=1,
        num_computed_tokens=4,
    )

    batch = sched.schedule()
    assert batch is not None
    assert batch.is_prefill
    assert batch.to_prefill == []
    assert batch.preempted == [req]
    assert req.status == RequestStatus.WAITING
    assert req.num_preemptions == 1
    assert req.num_computed_tokens == 0
    assert req.num_generated_tokens == 0
    assert req.prefix_tokens_saved == 0
    assert req.block_table == []
    assert req in sched.waiting
    assert req not in sched.running


def test_admission_allocation_failure_unlocks_prefix_handle_and_keeps_request_waiting() -> (
    None
):
    cm = StudentCacheManager(num_blocks=1, block_size=4)
    shared_prefix = [30, 31, 32, 33]

    cached_block_ids = _allocate_or_fail(cm, 1)
    cm.insert_prefix(shared_prefix, cached_block_ids)
    cm.free(cached_block_ids)  # keep as cached-only (evictable when unlocked)
    cached_block = cached_block_ids[0]

    _, sched = _make_scheduler(
        cache_manager=cm,
        num_blocks=1,
        block_size=4,
        max_seqs=1,
        token_budget=8,
        prefill_chunk=8,
        enable_prefix_caching=True,
    )
    req = Request(
        request_id=21,
        arrival_step=0,
        prompt_tokens=shared_prefix + [99, 100],  # requires one extra block
        max_new_tokens=2,
    )
    sched.add(req)

    batch = sched.schedule()
    assert batch is not None
    assert batch.is_prefill
    assert batch.newly_admitted == []
    assert req.status == RequestStatus.WAITING
    assert req.cache_handle is None
    assert len(sched.waiting) == 1
    assert len(sched.running) == 0
    # lock() should be rolled back via unlock() on allocation failure.
    assert cm.ref_counts[cached_block] == 1


def test_full_prefix_cache_hit_admits_without_prefill_compute() -> None:
    cm = StudentCacheManager(num_blocks=4, block_size=4)
    prompt = [41, 42, 43, 44, 45, 46, 47, 48]

    cached_block_ids = _allocate_or_fail(cm, 2)
    cm.insert_prefix(prompt, cached_block_ids)
    cm.free(cached_block_ids)

    _, sched = _make_scheduler(
        cache_manager=cm,
        num_blocks=4,
        block_size=4,
        max_seqs=1,
        token_budget=8,
        prefill_chunk=8,
        enable_prefix_caching=True,
    )
    req = Request(request_id=22, arrival_step=0, prompt_tokens=prompt, max_new_tokens=2)
    sched.add(req)

    batch = sched.schedule()
    assert batch is not None
    assert batch.is_prefill
    assert req in batch.newly_admitted
    assert batch.to_prefill == []
    assert req.status == RequestStatus.RUNNING
    assert req.num_computed_tokens == req.prompt_len
    assert req.prefix_tokens_saved == req.prompt_len
    assert req.cache_handle is not None


def test_prefill_admission_respects_max_seqs_and_fcfs_order() -> None:
    _, sched = _make_scheduler(
        num_blocks=8, block_size=4, max_seqs=1, token_budget=8, prefill_chunk=4
    )
    first = Request(30, 0, [1, 2, 3, 4], 1)
    second = Request(31, 0, [5, 6, 7, 8], 1)
    sched.add(first)
    sched.add(second)

    batch = sched.schedule()
    assert batch is not None
    assert batch.is_prefill
    assert batch.newly_admitted == [first]
    assert len(sched.running) == 1
    assert sched.running[0] is first
    assert len(sched.waiting) == 1
    assert sched.waiting[0] is second


def test_decode_first_uses_prefill_when_no_decode_ready_requests() -> None:
    _, sched = _make_scheduler(
        num_blocks=8,
        block_size=4,
        max_seqs=1,
        token_budget=4,
        prefill_chunk=4,
        scheduling_policy=SchedulingPolicy.DECODE_FIRST,
    )
    req = Request(40, 0, [1, 2, 3, 4], 2)
    sched.add(req)

    batch = sched.schedule()
    assert batch is not None
    assert batch.is_prefill
    assert req in batch.newly_admitted
    assert batch.to_decode == []


def test__schedule_prefill_continues_running_then_admits_waiting_with_remaining_budget() -> (
    None
):
    cm, sched = _make_scheduler(
        num_blocks=10, block_size=4, max_seqs=2, token_budget=6, prefill_chunk=4
    )

    running_req = Request(50, 0, [1, 2, 3, 4, 5, 6], 1)
    _attach_running_request(
        sched=sched,
        cm=cm,
        req=running_req,
        block_count=2,  # admission policy: running prefill requests own full prompt blocks
        num_computed_tokens=4,
    )

    waiting_req = Request(51, 0, [9, 8, 7, 6], 1)
    sched.add(waiting_req)

    batch = sched._schedule_prefill()
    assert batch.is_prefill
    print("BATCH TO=PREFIIl:", batch.to_prefill)
    # Running request consumes first: remaining=2 so it uses chunk=2.
    assert batch.to_prefill[0] == (running_req, 2)
    # Admission gets the leftover budget (6 - 2 = 4) and chunk cap=4.
    assert batch.to_prefill[1] == (waiting_req, 4)
    assert batch.newly_admitted == [waiting_req]
    assert waiting_req.status == RequestStatus.RUNNING
    assert len(sched.waiting) == 0
    assert waiting_req in sched.running


def test__schedule_prefill_admission_failure_breaks_loop_and_preserves_fcfs() -> None:
    _, sched = _make_scheduler(
        num_blocks=1, block_size=4, max_seqs=2, token_budget=8, prefill_chunk=8
    )

    first = Request(60, 0, [1, 2, 3, 4, 5], 1)  # needs 2 blocks -> will fail
    second = Request(61, 0, [9, 8, 7, 6], 1)  # would fit, but must not bypass FCFS
    sched.add(first)
    sched.add(second)

    batch = sched._schedule_prefill()
    assert batch.is_prefill
    assert batch.newly_admitted == []
    assert batch.to_prefill == []
    # Admission failure should stop further admissions this step (no FCFS bypass).
    assert list(sched.waiting) == [first, second]
    assert sched.running == []


def test__schedule_decode_only_schedules_decode_ready_and_skips_prefilling() -> None:
    cm, sched = _make_scheduler(
        num_blocks=8, block_size=4, max_seqs=3, token_budget=8, prefill_chunk=4
    )

    decode_ready = Request(70, 0, [1, 2, 3, 4], 3)
    _attach_running_request(
        sched=sched,
        cm=cm,
        req=decode_ready,
        block_count=1,
        num_computed_tokens=decode_ready.prompt_len,
    )

    still_prefilling = Request(71, 0, [5, 6, 7, 8, 9], 3)
    _attach_running_request(
        sched=sched,
        cm=cm,
        req=still_prefilling,
        block_count=2,
        num_computed_tokens=2,
    )

    batch = sched._schedule_decode()
    assert batch.is_decode
    assert batch.to_decode == [decode_ready]
    assert still_prefilling not in batch.to_decode


def test__schedule_decode_includes_requests_that_need_no_new_block() -> None:
    """Most decode steps don't cross a block boundary. Those requests must
    still appear in to_decode — the allocation check only determines whether
    a new block is needed, not whether the request is included."""
    cm, sched = _make_scheduler(
        num_blocks=8, block_size=4, max_seqs=2, token_budget=8, prefill_chunk=4
    )

    req = Request(90, 0, [1, 2, 3, 4], 5)
    _attach_running_request(
        sched=sched,
        cm=cm,
        req=req,
        block_count=2,
        num_computed_tokens=4,
        num_generated_tokens=2,  # total so far: 4+2=6, next=7 → blocks_for(7)=2, already has 2
    )

    batch = sched._schedule_decode()
    assert batch.is_decode
    assert req in batch.to_decode, (
        "decode-ready request must be in batch even when no new block is needed"
    )


def test__schedule_decode_excludes_prefilling_request_that_would_cross_block_boundary() -> (
    None
):
    """A still-prefilling request must be skipped by decode, even if the
    block-count arithmetic would suggest it needs a new block."""
    cm, sched = _make_scheduler(
        num_blocks=8, block_size=4, max_seqs=2, token_budget=8, prefill_chunk=4
    )

    prefilling_req = Request(91, 0, [1, 2, 3, 4, 5, 6, 7, 8], 3)
    _attach_running_request(
        sched=sched,
        cm=cm,
        req=prefilling_req,
        block_count=1,
        num_computed_tokens=4,  # prompt_len=8, still prefilling
    )

    batch = sched._schedule_decode()
    assert batch.is_decode
    assert prefilling_req not in batch.to_decode
    assert len(batch.to_decode) == 0


def test__schedule_decode_does_not_skip_requests_after_preemption() -> None:
    """When one request is preempted (removing it from self.running), the
    loop must still visit subsequent requests. This requires iterating over
    a snapshot copy, not the live list."""
    cm, sched = _make_scheduler(
        num_blocks=2, block_size=4, max_seqs=2, token_budget=8, prefill_chunk=4
    )

    # Both decode-ready, each with 1 block, both need a new block.
    req_a = Request(92, 0, [1, 2, 3, 4], 3)
    _attach_running_request(
        sched=sched,
        cm=cm,
        req=req_a,
        block_count=1,
        num_computed_tokens=4,
    )
    req_b = Request(93, 0, [5, 6, 7, 8], 3)
    _attach_running_request(
        sched=sched,
        cm=cm,
        req=req_b,
        block_count=1,
        num_computed_tokens=4,
    )

    assert cm.num_free_blocks == 0

    batch = sched._schedule_decode()
    assert batch.is_decode
    # A is preempted (no free blocks), freeing its block for B.
    assert req_a in batch.preempted
    assert req_b in batch.to_decode, (
        "request B must not be skipped after A's preemption"
    )


def test_prefill_admission_with_cache_hit_does_not_alias_cache_block_list() -> None:
    """When admission builds block_table from a cache hit, it must create
    a new list — not alias the cache's internal block list. Otherwise,
    extending block_table (for unique suffix blocks) corrupts the cache."""
    cm = StudentCacheManager(num_blocks=8, block_size=4)
    shared_prefix = [20, 21, 22, 23, 24, 25, 26, 27]

    cached_block_ids = _allocate_or_fail(cm, 2)
    cm.insert_prefix(shared_prefix, cached_block_ids)
    cm.free(cached_block_ids)

    original_cache_blocks = cm.cache_entries[tuple(shared_prefix)]

    _, sched = _make_scheduler(
        cache_manager=cm,
        num_blocks=8,
        block_size=4,
        max_seqs=2,
        token_budget=8,
        prefill_chunk=8,
        enable_prefix_caching=True,
    )
    req = Request(
        request_id=100,
        arrival_step=0,
        prompt_tokens=shared_prefix + [101, 102],
        max_new_tokens=2,
    )
    sched.add(req)
    batch = sched.schedule()
    assert batch is not None
    assert req in batch.newly_admitted

    current_cache_blocks = cm.cache_entries[tuple(shared_prefix)]
    assert current_cache_blocks == original_cache_blocks, (
        "extending req.block_table must not mutate the cache's stored block list"
    )


def test_waiting_queue_remains_deque_after_schedule() -> None:
    """self.waiting is a deque; _preempt relies on appendleft. Removing
    admitted requests during scheduling must not convert it to a plain list."""
    _, sched = _make_scheduler(
        num_blocks=8, block_size=4, max_seqs=1, token_budget=4, prefill_chunk=4
    )

    req1 = Request(100, 0, [1, 2, 3, 4], 2)
    req2 = Request(101, 0, [5, 6, 7, 8], 2)
    sched.add(req1)
    sched.add(req2)

    batch = sched.schedule()
    assert batch is not None
    assert req1 in batch.newly_admitted

    assert hasattr(sched.waiting, "appendleft"), (
        "waiting queue must remain a deque after schedule (needed by _preempt)"
    )


def test__schedule_prefill_continues_after_preempting_running_request() -> None:
    """When a running prefill request is preempted (under-allocated blocks),
    the loop must continue to the next running request — not break."""
    cm, sched = _make_scheduler(
        num_blocks=3, block_size=4, max_seqs=3, token_budget=8, prefill_chunk=4
    )

    # A: under-allocated → will be preempted.
    req_a = Request(110, 0, [1, 2, 3, 4, 5, 6, 7, 8], 1)
    _attach_running_request(
        sched=sched,
        cm=cm,
        req=req_a,
        block_count=1,
        num_computed_tokens=4,
    )

    # B: properly allocated → should still get a prefill chunk.
    req_b = Request(111, 0, [9, 10, 11, 12, 13, 14], 1)
    _attach_running_request(
        sched=sched,
        cm=cm,
        req=req_b,
        block_count=2,
        num_computed_tokens=4,
    )

    batch = sched._schedule_prefill()
    assert batch.is_prefill
    assert req_a in batch.preempted
    assert any(r is req_b for r, _ in batch.to_prefill), (
        "running prefill must continue to next request after preempting one"
    )


def test_schedule_falls_through_to_decode_when_prefill_produces_nothing() -> None:
    """When _schedule_prefill yields an empty batch (e.g. admission failure)
    but decode-ready running requests exist, schedule() must fall through
    to _schedule_decode rather than returning a useless empty prefill batch."""
    cm, sched = _make_scheduler(
        num_blocks=2,
        block_size=4,
        max_seqs=2,
        token_budget=8,
        prefill_chunk=4,
        scheduling_policy=SchedulingPolicy.PREFILL_FIRST,
    )

    # Decode-ready request owns 1 block; will need 1 more at the boundary.
    decode_req = Request(120, 0, [1, 2, 3, 4], 5)
    _attach_running_request(
        sched=sched,
        cm=cm,
        req=decode_req,
        block_count=1,
        num_computed_tokens=4,
    )

    # Waiting request needs 2 blocks but only 1 free — cannot be admitted.
    waiting_req = Request(121, 0, [5, 6, 7, 8, 9], 2)
    sched.add(waiting_req)

    batch = sched.schedule()
    assert batch is not None
    assert batch.is_decode, (
        "schedule must fall through to decode when prefill produces no useful work"
    )
    assert decode_req in batch.to_decode


def test__schedule_decode_preempts_on_block_boundary_when_allocation_fails() -> None:
    cm, sched = _make_scheduler(
        num_blocks=1, block_size=4, max_seqs=1, token_budget=8, prefill_chunk=4
    )
    req = Request(80, 0, [1, 2, 3, 4], 3)
    _attach_running_request(
        sched=sched,
        cm=cm,
        req=req,
        block_count=1,
        num_computed_tokens=4,
        num_generated_tokens=0,  # next decode token crosses into a new block
    )

    batch = sched._schedule_decode()
    assert batch.is_decode
    assert batch.to_decode == []
    assert batch.preempted == [req]
    assert req.status == RequestStatus.WAITING
    assert req.num_preemptions == 1
    assert req.block_table == []
    assert req in sched.waiting
    assert req not in sched.running


def test_full_prefix_match_does_not_leave_request_in_waiting_queue() -> None:
    """When the entire prompt is covered by the prefix cache the request has
    nothing to prefill, but it must still be admitted to `running` AND removed
    from `waiting`. Otherwise the next _schedule_prefill() iterates over a
    request that is no longer prefilling and trips `assert req.is_prefilling`."""
    cm = StudentCacheManager(num_blocks=8, block_size=2)

    # Pre-populate the cache with a prompt-sized prefix, then drop request
    # ownership so the blocks are cached-only.
    blocks = cm.allocate(2)
    assert blocks is not None
    cm.insert_prefix(tokens=[10, 11, 12, 13], block_ids=blocks)
    cm.free(blocks)

    _, sched = _make_scheduler(
        cache_manager=cm,
        block_size=2,
        max_seqs=4,
        token_budget=100,
        prefill_chunk=50,
        enable_prefix_caching=True,
        scheduling_policy=SchedulingPolicy.PREFILL_FIRST,
    )

    req = Request(
        request_id=1,
        arrival_step=0,
        prompt_tokens=[10, 11, 12, 13],
        max_new_tokens=3,
    )
    sched.add(req)

    batch = sched.schedule()
    assert batch is not None

    assert req in sched.running, "fully-cached request must be admitted to running"
    assert req in batch.newly_admitted
    assert req not in sched.waiting, (
        "fully-cached request must be removed from waiting; otherwise the next "
        "schedule() will re-process a non-prefilling request and assert."
    )

    # And the follow-up schedule() must not crash on the leaked waiter.
    sched.schedule()


def test_full_prefix_match_does_not_alias_cache_internal_block_list() -> None:
    """On a full-prompt cache hit the scheduler must still build a fresh
    block_table list. Aliasing `cache._cache[key]` means future operations
    (like decode-phase `block_table.extend(...)`) silently mutate the cache's
    stored block list and corrupt future prefix lookups."""
    cm = StudentCacheManager(num_blocks=8, block_size=2)

    blocks = cm.allocate(2)
    assert blocks is not None
    cm.insert_prefix(tokens=[10, 11, 12, 13], block_ids=blocks)
    cm.free(blocks)

    _, sched = _make_scheduler(
        cache_manager=cm,
        block_size=2,
        max_seqs=4,
        token_budget=100,
        prefill_chunk=50,
        enable_prefix_caching=True,
        scheduling_policy=SchedulingPolicy.PREFILL_FIRST,
    )

    req = Request(
        request_id=1,
        arrival_step=0,
        prompt_tokens=[10, 11, 12, 13],
        max_new_tokens=3,
    )
    sched.add(req)
    sched.schedule()

    cached_blocks = cm.cache_entries[(10, 11, 12, 13)]
    assert req.block_table == cached_blocks, "request starts with the cached blocks"
    assert req.block_table is not cm._cache[(10, 11, 12, 13)], (
        "block_table must be a fresh list, not an alias of the cache's internal list"
    )

    # Concrete corruption check: simulating a decode-time append on the
    # request's block_table must NOT mutate the cache entry.
    req.block_table.append(999)
    assert 999 not in cm.cache_entries[(10, 11, 12, 13)], (
        "appending to req.block_table leaked into the cache entry"
    )
