"""Small, focused correctness tests for CacheManager."""

from __future__ import annotations

from hw3_task import CacheManager as StudentCacheManager


def test_allocate_returns_none_when_capacity_cannot_be_freed() -> None:
    cm = StudentCacheManager(num_blocks=2, block_size=4)
    claimed = cm.allocate(2)
    assert claimed is not None

    # All blocks are live (ref > 0), so eviction cannot help.
    assert cm.allocate(1) is None


def test_allocate_and_free_roundtrip() -> None:
    cm = StudentCacheManager(num_blocks=5, block_size=4)

    assert cm.num_free_blocks == 5
    claimed = cm.allocate(3)
    assert claimed is not None
    assert len(claimed) == 3
    assert cm.num_free_blocks == 2

    # In production this happens when a request finishes (or is preempted):
    # the scheduler/engine releases the request-owned KV blocks back to the pool.
    cm.free(claimed)
    assert cm.num_free_blocks == 5


def test_match_prefix_hits_only_complete_blocks() -> None:
    cm = StudentCacheManager(num_blocks=6, block_size=4)

    blocks = cm.allocate(2)
    assert blocks is not None

    # Only the first 4 tokens form one full block.
    tokens = [1, 2, 3, 4, 5, 6]
    # Typical lifecycle: a completed request inserts its prompt prefix so future
    # requests can reuse those blocks via match_prefix.
    cm.insert_prefix(tokens=tokens, block_ids=blocks)
    # Then the same request releases ownership of those blocks; they remain
    # cache-owned/evictable instead of staying pinned by the finished request.
    cm.free(blocks)

    hit = cm.match_prefix(tokens + [9, 10])
    assert hit.num_matched_tokens == 4
    assert len(hit.matched_blocks) == 1

    miss = cm.match_prefix([100, 101, 102, 103])
    assert miss.num_matched_tokens == 0
    assert miss.matched_blocks == []


def test_insert_prefix_updates_cache_ref_and_ref_in_simple_steps() -> None:
    cm = StudentCacheManager(num_blocks=4, block_size=2)
    blocks = cm.allocate(2)
    assert blocks is not None
    b0, b1 = blocks

    # Two complete blocks create two cache keys:
    # (10,11) -> [b0], (10,11,12,13) -> [b0,b1]
    cm.insert_prefix(tokens=[10, 11, 12, 13], block_ids=blocks)

    # cache_ref counts cache-entry overlap; ref gets only one cache-ownership bump.
    assert cm.cache_ref_counts[b0] == 2
    assert cm.cache_ref_counts[b1] == 1
    assert cm.ref_counts[b0] == 2
    assert cm.ref_counts[b1] == 2

    # After request free(), blocks become cached-only: ref=1 while cache_ref stays.
    cm.free(blocks)
    assert cm.ref_counts[b0] == 1
    assert cm.ref_counts[b1] == 1


def test_cache_entries_show_complete_prefix_keys() -> None:
    cm = StudentCacheManager(num_blocks=4, block_size=2)
    blocks = cm.allocate(2)
    assert blocks is not None
    b0, b1 = blocks

    cm.insert_prefix(tokens=[10, 11, 12, 13], block_ids=blocks)

    # Two complete-block prefixes should be cached.
    assert cm.cache_entries[(10, 11)] == [b0]
    assert cm.cache_entries[(10, 11, 12, 13)] == [b0, b1]


def test_lru_updates_when_a_cached_prefix_is_hit() -> None:
    cm = StudentCacheManager(num_blocks=4, block_size=2)

    a = cm.allocate(1)
    b = cm.allocate(1)
    assert a is not None and b is not None

    cm.insert_prefix(tokens=[1, 2], block_ids=a)
    cm.free(a)
    cm.insert_prefix(tokens=[3, 4], block_ids=b)
    cm.free(b)

    # Initial insertion order: A then B.
    assert cm.lru_keys == [(1, 2), (3, 4)]

    # Matching A should move A to most-recently used.
    hit = cm.match_prefix([1, 2, 9])
    assert hit.num_matched_tokens == 2
    assert cm.lru_keys == [(3, 4), (1, 2)]


def test_lock_unlock_changes_only_ref_not_cache_ref() -> None:
    cm = StudentCacheManager(num_blocks=3, block_size=2)
    blocks = cm.allocate(2)
    assert blocks is not None

    cm.insert_prefix(tokens=[1, 2, 3, 4], block_ids=blocks)
    cm.free(blocks)

    handle = cm.match_prefix([1, 2, 3, 4, 99])
    cache_ref_before = cm.cache_ref_counts[:]
    ref_before = cm.ref_counts[:]

    cm.lock(handle)
    assert cm.cache_ref_counts == cache_ref_before
    assert sum(cm.ref_counts) == sum(ref_before) + len(handle.matched_blocks)

    cm.unlock(handle)
    assert cm.cache_ref_counts == cache_ref_before
    assert cm.ref_counts == ref_before


def test_lock_prevents_eviction_until_unlock() -> None:
    cm = StudentCacheManager(num_blocks=2, block_size=4)

    blocks = cm.allocate(2)
    assert blocks is not None
    tokens = [10, 11, 12, 13, 14, 15, 16, 17]
    cm.insert_prefix(tokens=tokens, block_ids=blocks)
    # Simulates request teardown: drop request ownership, keep cache ownership.
    cm.free(blocks)

    handle = cm.match_prefix(tokens + [999])
    assert handle.num_matched_tokens == 8
    # New request that hits the cache must lock matched blocks before use.
    cm.lock(handle)

    # No free blocks and cached blocks are locked => cannot evict.
    assert cm.allocate(1) is None

    # When that request is done, unlock so blocks become evictable again.
    cm.unlock(handle)
    assert cm.allocate(1) is not None


def test_direct_evict_skips_locked_entries() -> None:
    cm = StudentCacheManager(num_blocks=2, block_size=4)
    blocks = cm.allocate(2)
    assert blocks is not None

    tokens = [10, 11, 12, 13, 14, 15, 16, 17]
    short_key = tuple(tokens[:4])
    long_key = tuple(tokens)

    cm.insert_prefix(tokens=tokens, block_ids=blocks)
    cm.free(blocks)
    handle = cm.match_prefix(tokens + [999])
    cm.lock(handle)

    cm._evict_blocks_from_kv_cache(1)

    # Locked blocks are not evictable.
    assert cm.num_free_blocks == 0
    assert short_key in cm.cache_entries
    assert long_key in cm.cache_entries


def test_overlap_eviction_may_need_multiple_keys_for_one_block() -> None:
    cm = StudentCacheManager(num_blocks=2, block_size=4)
    blocks = cm.allocate(2)
    assert blocks is not None

    tokens = [10, 11, 12, 13, 14, 15, 16, 17]
    short_key = tuple(tokens[:4])
    long_key = tuple(tokens)

    cm.insert_prefix(tokens=tokens, block_ids=blocks)
    cm.free(blocks)

    # With overlap, evicting only the shorter key is not enough to free a block.
    cm._evict_blocks_from_kv_cache(1)

    assert cm.num_free_blocks >= 1
    assert short_key not in cm.cache_entries
    assert long_key not in cm.cache_entries


def test_evict_respects_requested_amount_not_all_evictable() -> None:
    cm = StudentCacheManager(num_blocks=3, block_size=2)

    a = cm.allocate(1)
    b = cm.allocate(1)
    c = cm.allocate(1)
    assert a is not None and b is not None and c is not None

    # Create three independent one-block cache entries in LRU order: A, B, C.
    cm.insert_prefix(tokens=[1, 2], block_ids=a)
    cm.free(a)
    cm.insert_prefix(tokens=[3, 4], block_ids=b)
    cm.free(b)
    cm.insert_prefix(tokens=[5, 6], block_ids=c)
    cm.free(c)

    assert cm.lru_keys == [(1, 2), (3, 4), (5, 6)]

    # Ask to reclaim only one block.
    cm._evict_blocks_from_kv_cache(1)

    # Should evict only the oldest entry, not all evictable entries.
    assert cm.num_free_blocks == 1
    assert cm.match_prefix([1, 2, 9]).num_matched_tokens == 0
    assert cm.match_prefix([3, 4, 9]).num_matched_tokens == 2
    assert cm.match_prefix([5, 6, 9]).num_matched_tokens == 2


def test_eviction_removes_cache_ownership_for_evicted_blocks() -> None:
    cm = StudentCacheManager(num_blocks=3, block_size=2)
    blocks = cm.allocate(2)
    assert blocks is not None
    b0, b1 = blocks

    cm.insert_prefix(tokens=[7, 8, 9, 10], block_ids=blocks)
    cm.free(blocks)

    # Requires eviction because only one free block exists.
    assert cm.allocate(2) is not None

    # Both cached entries were removed; no cache ownership remains.
    assert cm.cache_ref_counts[b0] == 0
    assert cm.cache_ref_counts[b1] == 0


def test_insert_prefix_is_idempotent_for_existing_keys() -> None:
    """Calling insert_prefix twice with the same token prefix must not
    create duplicate LRU entries or inflate cache_ref / ref counts."""
    cm = StudentCacheManager(num_blocks=4, block_size=2)
    blocks = cm.allocate(2)
    assert blocks is not None
    b0, b1 = blocks

    tokens = [10, 11, 12, 13]
    cm.insert_prefix(tokens=tokens, block_ids=blocks)

    lru_snapshot = cm.lru_keys[:]
    cache_ref_snapshot = cm.cache_ref_counts[:]
    ref_snapshot = cm.ref_counts[:]

    # Second insert with the same prefix (simulates a second request finishing
    # with the same shared prompt).
    cm.insert_prefix(tokens=tokens, block_ids=blocks)

    assert cm.lru_keys == lru_snapshot, "LRU must not gain duplicate entries"
    assert cm.cache_ref_counts == cache_ref_snapshot, (
        "cache_ref must not be double-counted"
    )
    assert cm.ref_counts == ref_snapshot, (
        "ref must not change for already-cached prefixes"
    )


def test_evict_stops_after_enough_blocks_freed_without_over_evicting() -> None:
    """When evicting overlapping entries frees more blocks than requested,
    the eviction loop must stop and not evict additional entries."""
    cm = StudentCacheManager(num_blocks=3, block_size=2)

    # Two-block overlapping entry: keys (1,2) and (1,2,3,4) share block 0.
    blocks_ab = cm.allocate(2)
    assert blocks_ab is not None
    cm.insert_prefix(tokens=[1, 2, 3, 4], block_ids=blocks_ab)
    cm.free(blocks_ab)

    # Independent one-block entry.
    blocks_c = cm.allocate(1)
    assert blocks_c is not None
    cm.insert_prefix(tokens=[5, 6], block_ids=blocks_c)
    cm.free(blocks_c)

    # Ask to reclaim 1 block. Evicting the overlapping entries for (1,2,3,4)
    # frees 2 blocks (overshooting by 1). The independent entry must survive.
    cm._evict_blocks_from_kv_cache(1)

    assert cm.num_free_blocks >= 1
    assert cm.match_prefix([5, 6, 99]).num_matched_tokens == 2, (
        "independent entry must not be evicted when enough blocks were already freed"
    )


def test_match_prefix_only_refreshes_longest_match_in_lru() -> None:
    """When a multi-block prefix matches, only the longest-match key should
    move to MRU position. Shorter sub-prefix keys keep their LRU rank."""
    cm = StudentCacheManager(num_blocks=4, block_size=2)

    # Two-block cached prefix produces keys (1,2) and (1,2,3,4) in the LRU.
    blocks = cm.allocate(2)
    assert blocks is not None
    cm.insert_prefix(tokens=[1, 2, 3, 4], block_ids=blocks)
    cm.free(blocks)

    # Independent entry after the two-block prefix.
    other = cm.allocate(1)
    assert other is not None
    cm.insert_prefix(tokens=[5, 6], block_ids=other)
    cm.free(other)

    # LRU before match: [(1,2), (1,2,3,4), (5,6)]
    assert cm.lru_keys == [(1, 2), (1, 2, 3, 4), (5, 6)]

    hit = cm.match_prefix([1, 2, 3, 4, 99])
    assert hit.num_matched_tokens == 4

    # Only (1,2,3,4) should be refreshed, not (1,2).
    # Expected: [(1,2), (5,6), (1,2,3,4)]
    assert cm.lru_keys[0] == (1, 2), (
        "sub-prefix key must not be refreshed — only the longest match moves to MRU"
    )


def test_evicts_least_recently_used_prefix() -> None:
    cm = StudentCacheManager(num_blocks=3, block_size=2)

    tokens_a = [1, 2]
    tokens_b = [3, 4]

    a = cm.allocate(1)
    b = cm.allocate(1)
    assert a is not None and b is not None

    # Two completed requests leave behind two one-block cached prefixes.
    cm.insert_prefix(tokens=tokens_a, block_ids=a)
    cm.free(a)
    cm.insert_prefix(tokens=tokens_b, block_ids=b)
    cm.free(b)

    # Touch B to make it most-recently used.
    assert cm.match_prefix(tokens_b + [99]).num_matched_tokens == 2

    # Needs one more block than free pool has; should evict A first.
    claimed = cm.allocate(2)
    assert claimed is not None
    assert len(claimed) == 2

    assert cm.match_prefix(tokens_a + [7]).num_matched_tokens == 0
    assert cm.match_prefix(tokens_b + [8]).num_matched_tokens == 2


def test_match_prefix_finds_longer_when_shorter_was_independently_evicted() -> None:
    """Shorter sub-prefix entries can be evicted independently of longer ones
    that share some of the same physical blocks. `match_prefix` must still
    return the longest cached prefix in that state (e.g. probe longest-first
    rather than stopping at the first miss)."""
    cm = StudentCacheManager(num_blocks=5, block_size=2)

    # Two-block prefix: produces cache entries (1,2) and (1,2,3,4).
    blocks_ab = cm.allocate(2)
    assert blocks_ab is not None
    cm.insert_prefix(tokens=[1, 2, 3, 4], block_ids=blocks_ab)
    cm.free(blocks_ab)

    # Independent one-block prefix. Needed so there's an entry we can actually
    # evict to satisfy a reclaim — evicting (1,2) by itself frees 0 physical
    # blocks because its block is still held by the longer (1,2,3,4) entry.
    blocks_c = cm.allocate(1)
    assert blocks_c is not None
    cm.insert_prefix(tokens=[5, 6], block_ids=blocks_c)
    cm.free(blocks_c)

    # Touch (1,2,3,4) so it moves to MRU; (1,2) stays at the LRU head.
    assert cm.match_prefix([1, 2, 3, 4, 99]).num_matched_tokens == 4
    assert cm.lru_keys == [(1, 2), (5, 6), (1, 2, 3, 4)]

    # Force eviction. 5 blocks total, 3 cache-held, 2 free — allocating 3
    # must reclaim at least 1 more block. The eviction walks LRU from the head:
    #   (1,2) → cache_ref goes 2→1, frees 0 blocks, loop continues
    #   (5,6) → cache_ref goes 1→0, frees 1 block, loop stops
    # After this, (1,2) is gone but (1,2,3,4) is still cached.
    new_blocks = cm.allocate(3)
    assert new_blocks is not None
    assert (1, 2, 3, 4) in cm.cache_entries, (
        "longer entry must survive; it was MRU and its blocks were not freed"
    )
    assert (1, 2) not in cm.cache_entries, (
        "precondition: shorter sub-prefix must have been evicted for this test"
    )

    # The real assertion: match_prefix must still find the 4-token match.
    hit = cm.match_prefix([1, 2, 3, 4, 99, 100])
    assert hit.num_matched_tokens == 4, (
        f"expected to match the longer cached prefix (1,2,3,4) even though "
        f"(1,2) was evicted, got num_matched_tokens={hit.num_matched_tokens}"
    )
    assert hit.matched_blocks == cm.cache_entries[(1, 2, 3, 4)]
