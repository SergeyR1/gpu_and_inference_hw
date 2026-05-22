# HW3: Mini Inference Engine *(optional, ungraded)*

## Goal

Build the memory-management and scheduling core of a tiny LLM inference engine.
You will implement:

- A paged KV-cache manager with block allocation, prefix reuse, reference counts, and LRU eviction.
- A continuous-batching scheduler that admits requests, runs prefill and decode batches, handles memory pressure, and records useful latency metrics.

This homework runs on CPU. The model is intentionally small and synthetic so the interesting work is the engine state machine, not GPU kernels or model quality.

**Grading:** this assignment is **optional and ungraded**. It is included for students who want to go beyond HW1 + HW2 and see how a real inference engine manages KV memory and schedules concurrent requests internally; nothing here counts toward the course total.

## What You Should Learn

Real inference engines are mostly about keeping expensive compute and scarce KV memory under control while many requests are in flight. This assignment focuses on three ideas that show up in systems such as vLLM, SGLang, TensorRT-LLM, and TGI:

- **Paged KV memory:** requests refer to fixed-size physical blocks through a per-request `block_table`, instead of owning one contiguous KV tensor.
- **Prefix caching:** completed prompt blocks can remain in cache and be reused by later requests with the same token prefix.
- **Continuous batching:** each engine step runs one homogeneous batch, but the set of active requests changes over time as new requests arrive, old requests finish, and memory pressure forces preemption.

The implementation is deliberately smaller than a production engine, but the ownership rules are real.

> **Recommended background:** Lecture 2 covers Paged Attention, Prefix Caching, and Continuous Batching — the three ideas this homework is built around. Watching it first will make the `CacheManager` and `Scheduler` contracts below easier to read.

## File Layout

- `hw3_task.py`: **your only file to edit**. Implement `CacheManager` and `Scheduler`, then answer the writeup questions at the bottom.
- `structs.py`: request, batch, scheduling-policy, and metrics dataclasses.
- `dummy_llm.py`: a tiny CPU model with explicit paged-KV read/write helpers.
- `engine_utils.py`: workload generation, plotting, constants, and stats.
- `test_cache_manager_correctness.py`: focused tests for allocation, prefix caching, reference counts, and eviction.
- `test_scheduler_correctness.py`: focused tests for batching, admission, preemption, and prefix-cache integration.
- `test_hw3_correctness.py`: smaller end-to-end smoke tests.

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```bash
python3 hw3_inference_engine/hw3_task.py
```

A correct implementation runs two synthetic workloads, each with and without prefix caching, and then compares prefill-first against decode-first scheduling. It saves plots in `hw3_inference_engine/results/`.

- **Prefill-heavy:** large prompts and shorter outputs. Prefix caching should save many prefill tokens, so TTFT and total steps improve substantially.
- **Decode-heavy:** shorter prompts and longer outputs. Prefix caching still helps, but decode work dominates, so the benefit is smaller.

Your exact numbers may differ by machine, but the qualitative pattern should be stable.

## System Overview

The engine loop is already provided in `MiniEngine.run()`. Each step has three roles:

1. Newly arrived requests are appended to the scheduler's waiting queue.
2. The scheduler returns either one prefill batch or one decode batch.
3. The model executes that batch, and the engine finalizes any requests that finished during the step.

`CacheManager`, `Scheduler`, and `DummyLLM` have separate responsibilities:

- `CacheManager` owns **physical block IDs**. It knows which blocks are free, cached, pinned by live requests, or evictable.
- `Scheduler` owns **request progress**. It decides when requests become running, which phase to run this step, and which block IDs each request may use.
- `DummyLLM` owns the actual **KV tensors**. It trusts each request's `block_table` and uses those physical block IDs to scatter and gather KV.

That separation is the core design. `CacheManager` never stores activations or tokens beyond cache keys; `DummyLLM` never decides who owns memory; `Scheduler` is the only component that moves requests between waiting, running, and done.

## Request Lifecycle

Every request starts in `WAITING`, becomes `RUNNING` when the scheduler admits it, and ends in `DONE` after it has generated `max_new_tokens`.

While running, a request is in one of two sub-states:

- **Prefill:** `num_computed_tokens < prompt_len`. The request still needs prompt KV to be computed, usually in chunks.
- **Decode:** `num_computed_tokens == prompt_len`. Each decode step generates one new token and may occasionally require one more KV block.

Prefill and decode are never mixed in the same `Batch`. Production engines keep these phases separate because they use different attention kernels and different batch shapes. The simplified CPU model mirrors that split.

Preemption in this homework means "free the request's KV blocks and restart it later from the waiting queue." It is intentionally blunt. Use it only when memory cannot be obtained from the free pool or by evicting unused cached prefixes.

## Task 1: `CacheManager`

Implement a single component that acts as:

- A block allocator for live request KV.
- A prefix cache for complete prompt blocks.
- An LRU evictor for cached blocks that are not pinned by live requests.

### Block Ownership Model

There are five pieces of state to keep consistent:

- `_free`: physical block IDs with no owner.
- `_ref[block_id]`: effective ownership count for that block.
- `_cache_ref[block_id]`: number of prefix-cache entries that mention that block.
- `_cache`: map from a complete-block token prefix to the physical blocks that store it.
- `_lru`: cache keys ordered from least recently used to most recently used.

The important distinction is that `_cache_ref` counts cache-entry overlap, while `_ref` answers "can this physical block be freed right now?"

A physical block can be in one of these states:


| State                          | Meaning                                                          |
| ------------------------------ | ---------------------------------------------------------------- |
| `_ref == 0`                    | Free block. It should appear in `_free`.                         |
| `_ref == 1`, `_cache_ref == 0` | Owned by one live request only.                                  |
| `_ref == 1`, `_cache_ref > 0`  | Cached only. It may be evicted by LRU.                           |
| `_ref >= 2`                    | Cached and pinned by at least one live request. Do not evict it. |


When the first cache entry starts referencing a block, the cache gains one ownership unit in `_ref`. Additional overlapping cache entries increase `_cache_ref`, but they should not keep incrementing `_ref`. When the last cache entry for a block is evicted, that one cache ownership unit is released.

### Prefix Cache Semantics

Only complete blocks are cacheable. With a block size of 16, a 40-token prompt can cache prefixes of length 16 and 32, not 40.

`match_prefix(tokens)` should return the longest cached complete-block prefix for the requested tokens. A match is only a lookup result; it does not make the blocks safe for a live request yet. The scheduler must call `lock(handle)` before using matched blocks and `unlock(handle)` when the request is done or preempted.

`insert_prefix(tokens, block_ids)` is called when a request finishes, before the request releases its own ownership. This lets the cache adopt complete prompt blocks cleanly. Re-inserting a prefix that already exists should be a no-op for that key: duplicate cache keys should not create duplicate LRU entries or inflate reference counts.

### Eviction Semantics

Allocation should first try the free list. If there are not enough free blocks, it should evict least-recently-used cache entries whose blocks are not pinned by live requests. If eviction still cannot reclaim enough blocks, allocation should return `None`; the scheduler decides whether that failure causes admission to wait or a running request to be preempted.

Two details are worth keeping in mind:

- Cached prefixes can overlap. Evicting one cache key may free zero blocks because a longer or shorter cache key still references the same physical block.
- LRU tracks cache entries, not physical blocks. Refresh the entry that actually matched a request; do not treat every sub-prefix as recently used just because a longer prefix matched.

## Task 2: `Scheduler`

Implement the control plane that turns waiting and running requests into one batch per step.

The scheduler must preserve these contracts:

- A returned batch is phase-pure: prefill or decode, never both.
- `max_seqs` limits how many requests may be running at once.
- `token_budget` limits how much prefill work can be scheduled in one step.
- Waiting requests are admitted in first-come-first-served order.
- A running prefill request should already own enough prompt blocks for its full prompt. Decode requests allocate at most when they cross a block boundary.
- Prefix-cache hits reduce prefill work, but matched blocks must be locked while the request is live.
- Memory pressure is handled in this order: use free blocks, let `CacheManager.allocate()` evict cached-only prefixes, then preempt a running request if allocation still fails.

### Phase Selection

The scheduler supports two policies:

- `PREFILL_FIRST`: prefer prefill work whenever there is waiting or in-flight prefill work; run decode when there is no useful prefill work available.
- `DECODE_FIRST`: prefer decode-ready running requests; run prefill when no request is ready to decode.

"Useful work" matters. If a prefill attempt cannot admit or continue any request, but decode-ready requests exist, the step should still run decode rather than returning an empty prefill batch.

### Admission

Admission is the transition from `WAITING` to `RUNNING`. On admission, initialize all scheduler-owned request fields consistently:

- `status`
- `block_table`
- `num_computed_tokens`
- `prefix_tokens_saved`
- `cache_handle`
- `first_scheduled_step`

The waiting queue is FIFO. If the request at the front cannot be admitted because there is not enough memory, leave it at the front and stop admitting for that step. Do not skip over it to admit a smaller later request.

A full-prefix cache hit is still an admission. The request may have no prefill tokens left to schedule, but it must be removed from `waiting`, added to `running`, and made ready for a future decode step.

### Decode

Decode-ready requests generate one token per decode batch. Most decode steps do not need a new block; only positions that cross a block boundary do. The allocation check should decide whether to extend the request's `block_table`, not whether the request belongs in the decode batch.

If allocation fails for a running decode request, preempt that request and keep considering the rest of the running requests for the same step.

## Testing

Work test-first, but read the model first. A good sequence is:

1. Read `structs.py`, then the `CacheManager` and `Scheduler` TODOs in `hw3_task.py`.
2. Read the first few tests in the suite you are working on to understand the intended public behavior.
3. Implement one small piece, run the relevant focused tests, and repeat.

Run the suites from the repository root:

```bash
python -m pytest hw3_inference_engine/test_cache_manager_correctness.py -v
python -m pytest hw3_inference_engine/test_scheduler_correctness.py -v
python -m pytest hw3_inference_engine/test_hw3_correctness.py -v
```

The tests cover many edge cases, but they are not a substitute for maintaining the invariants above. If a fix makes one test pass by special-casing that test, it will usually break another part of the engine.

## Writeup

After your implementation runs, answer the four questions at the bottom of `hw3_task.py`. Use numbers from your own run where requested.