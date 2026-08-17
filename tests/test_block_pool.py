import random
import unittest

from infurnace.cache.block_pool import BlockPool, BlockPoolError


def _state_sets(pool):
    free = set(pool._free)
    active = set().union(*pool._active.values()) if pool._active else set()
    inflight = set().union(*pool._in_flight.values()) if pool._in_flight else set()
    dummies = set(range(pool.num_pages, pool.num_pages + pool.dummy_pages))
    return free, active, inflight, dummies


def _assert_invariants(pool, tc):
    free, active, inflight, dummies = _state_sets(pool)
    all_pages = set(range(pool.num_pages))
    tc.assertTrue(free.isdisjoint(active), "free page is active")
    tc.assertTrue(free.isdisjoint(inflight), "free page is in flight")
    tc.assertEqual(free | active | inflight, all_pages, "page state partition is incomplete")
    tc.assertTrue(dummies.isdisjoint(free | active | inflight), "dummy page was allocated")
    for page in all_pages:
        refs = pool._refs.get(page, 0)
        if page in free:
            tc.assertEqual(refs, 0, f"free page {page} has refs")
        else:
            tc.assertGreaterEqual(refs, 1, f"held page {page} has no refs")


class TestBlockPool(unittest.TestCase):
    def test_alloc_release_cycle(self):
        pool = BlockPool(num_pages=8, page_size=16)
        self.assertEqual(pool.alloc("a", 3), (0, 1, 2))
        self.assertEqual(pool.num_free_pages, 5)
        pool.release("a")
        self.assertEqual(pool.num_free_pages, 8)
        self.assertFalse(pool.owns("a"))
        _assert_invariants(pool, self)

    def test_alloc_lowest_free_pages_first(self):
        pool = BlockPool(num_pages=8, page_size=16)
        pool.alloc("a", 2)
        pool.release("a")
        self.assertEqual(pool.alloc("b", 2), (0, 1))

    def test_alloc_atomic_rollback(self):
        pool = BlockPool(num_pages=4, page_size=16)
        self.assertIsNone(pool.alloc("a", 5))
        self.assertFalse(pool.owns("a"))
        self.assertEqual(pool.num_free_pages, 4)
        _assert_invariants(pool, self)

    def test_extend_and_atomicity(self):
        pool = BlockPool(num_pages=8, page_size=16)
        pool.alloc("a", 2)
        self.assertEqual(pool.extend("a", 2), (2, 3))
        self.assertEqual(pool.block_table("a"), (0, 1, 2, 3))
        self.assertIsNone(pool.extend("a", 5))
        self.assertEqual(pool.block_table("a"), (0, 1, 2, 3))
        self.assertEqual(pool.num_free_pages, 4)
        _assert_invariants(pool, self)

    def test_unique_ownership(self):
        pool = BlockPool(num_pages=8, page_size=16)
        pool.alloc("a", 3)
        pool.alloc("b", 3)
        a, b = set(pool.block_table("a")), set(pool.block_table("b"))
        self.assertTrue(a.isdisjoint(b))
        _assert_invariants(pool, self)

    def test_in_flight_delays_reclaim(self):
        pool = BlockPool(num_pages=8, page_size=16)
        pool.alloc("a", 2)
        pool.mark_in_flight("a")
        pool.release("a")
        self.assertEqual(pool.num_free_pages, 6)
        pool.complete_in_flight("a")
        self.assertEqual(pool.num_free_pages, 8)
        _assert_invariants(pool, self)

    def test_complete_before_release(self):
        pool = BlockPool(num_pages=8, page_size=16)
        pool.alloc("a", 2)
        pool.mark_in_flight("a")
        pool.complete_in_flight("a")
        self.assertEqual(pool.num_free_pages, 6)
        pool.release("a")
        self.assertEqual(pool.num_free_pages, 8)
        _assert_invariants(pool, self)

    def test_cancel_reclaims_except_in_flight(self):
        pool = BlockPool(num_pages=8, page_size=16)
        pool.alloc("a", 3)
        pool.mark_in_flight("a")
        pool.cancel("a")
        self.assertEqual(pool.num_free_pages, 5)
        self.assertFalse(pool.owns("a"))
        pool.complete_in_flight("a")
        self.assertEqual(pool.num_free_pages, 8)
        _assert_invariants(pool, self)

    def test_double_mark_is_idempotent(self):
        pool = BlockPool(num_pages=8, page_size=16)
        pool.alloc("a", 2)
        pool.mark_in_flight("a")
        pool.mark_in_flight("a")
        pool.release("a")
        self.assertEqual(pool.num_free_pages, 6)
        pool.complete_in_flight("a")
        self.assertEqual(pool.num_free_pages, 8)
        _assert_invariants(pool, self)

    def test_extend_while_in_flight_is_protected(self):
        pool = BlockPool(num_pages=8, page_size=16)
        pool.alloc("a", 2)
        pool.mark_in_flight("a")
        pool.extend("a", 1)
        pool.release("a")
        self.assertEqual(pool.num_free_pages, 5)
        pool.complete_in_flight("a")
        self.assertEqual(pool.num_free_pages, 8)
        _assert_invariants(pool, self)

    def test_dummy_pages_never_allocated(self):
        pool = BlockPool(num_pages=4, page_size=16, dummy_pages=2)
        self.assertEqual(pool.dummy_pages_start, 4)
        for i in range(4):
            self.assertEqual(pool.alloc(str(i), 1), (i,))
        self.assertTrue(pool.is_exhausted)
        self.assertIsNone(pool.alloc("z", 1))
        _assert_invariants(pool, self)

    def test_invalid_and_stale_ids(self):
        pool = BlockPool(num_pages=8, page_size=16)
        with self.assertRaises(BlockPoolError):
            pool.alloc("", 1)
        with self.assertRaises(BlockPoolError):
            pool.alloc("a", 0)
        with self.assertRaises(BlockPoolError):
            pool.extend("nope", 1)
        with self.assertRaises(BlockPoolError):
            pool.block_table("nope")
        pool.alloc("a", 1)
        with self.assertRaises(BlockPoolError):
            pool.alloc("a", 1)
        pool.release("a")
        pool.release("a")  # idempotent
        _assert_invariants(pool, self)

    def test_randomized_trace_preserves_invariants(self):
        rng = random.Random(7)
        pool = BlockPool(num_pages=64, page_size=16, dummy_pages=4)
        next_id = 0
        live = []
        for _ in range(500):
            op = rng.randrange(5)
            if op == 0 and len(live) < 8:
                rid = f"r{next_id}"
                next_id += 1
                if pool.alloc(rid, rng.randint(1, 4)) is not None:
                    live.append(rid)
            elif op == 1 and live:
                rid = rng.choice(live)
                pool.extend(rid, rng.randint(1, 3))
            elif op == 2 and live:
                rid = rng.choice(live)
                pool.mark_in_flight(rid)
            elif op == 3 and live:
                rid = rng.choice(live)
                pool.complete_in_flight(rid)
            elif op == 4 and live:
                rid = live.pop(rng.randrange(len(live)))
                pool.cancel(rid) if rng.random() < 0.5 else pool.release(rid)
            _assert_invariants(pool, self)


if __name__ == "__main__":
    unittest.main()