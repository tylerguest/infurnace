from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

class BlockPoolError(ValueError):
    """Block pool inputs or state do not satisfy the allocation contract."""

@dataclass
class BlockPool:
    """Logical page allocator for paged decode KV (Phase 5A).

    Pure-Python ownership bookkeeping with no tinygrad dependency. Manages the
    free page set, per-request block tables, reference counts, and separate
    active and in-flight ownership so pages are reclaimed only after active and
    in-flight ownership end. ``dummy_pages`` physical pages above ``num_pages``
    are reserved for padded decode rows and are never handed to a request; the
    physical pool tensor must be sized ``num_pages + dummy_pages``.

    Lifecycle:
      alloc(rid, n)           reserve n free pages atomically (all or none)
      extend(rid, n)          reserve n more pages for a growing sequence
      mark_in_flight(rid)     execution began; an extra reference is held
      complete_in_flight(rid) device work done; the execution reference drops
      release(rid) / cancel(rid)  terminal paths; ownership reference drops

    A page is free exactly when its reference count is zero. An active request
    holds one reference per page; each ``mark_in_flight`` adds an execution
    reference so submitted device work never reads recycled memory.
    """
    num_pages: int
    page_size: int
    dummy_pages: int = 1
    _free: Set[int] = field(init=False, default_factory=set)
    _block_tables: Dict[str, List[int]] = field(init=False, default_factory=dict)
    _active: Dict[str, Set[int]] = field(init=False, default_factory=dict)
    _in_flight: Dict[str, Set[int]] = field(init=False, default_factory=dict)
    _refs: Dict[int, int] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_pages < 1:
            raise BlockPoolError(f"num_pages must be >= 1, got {self.num_pages}")
        if self.page_size < 1:
            raise BlockPoolError(f"page_size must be >= 1, got {self.page_size}")
        if self.dummy_pages < 0:
            raise BlockPoolError(f"dummy_pages must be >= 0, got {self.dummy_pages}")
        self._free = set(range(self.num_pages))

    @property
    def dummy_pages_start(self) -> int:
        """First physical page index reserved for padded decode rows."""
        return self.num_pages

    def _check_id(self, request_id: str) -> None:
        if not isinstance(request_id, str) or not request_id:
            raise BlockPoolError(f"request_id must be a nonempty string, got {request_id!r}")

    def _reserve(self, request_id: str, pages: List[int]) -> None:
        self._free.difference_update(pages)
        self._block_tables[request_id].extend(pages)
        self._active[request_id].update(pages)
        for page in pages:
            self._refs[page] = self._refs.get(page, 0) + 1
        if request_id in self._in_flight:
            # A grow that happens mid-execution is read by the submitted work;
            # cover the new pages with the same execution reference.
            self._in_flight[request_id].update(pages)
            for page in pages:
                self._refs[page] += 1

    def _drop_refs(self, pages: Set[int]) -> None:
        for page in pages:
            self._refs[page] -= 1
            if self._refs[page] < 0:
                raise BlockPoolError(f"reference underflow for page {page}")
            if self._refs[page] == 0:
                if page in self._free:
                    raise BlockPoolError(f"double-free of page {page}")
                self._free.add(page)

    def alloc(self, request_id: str, num_pages: int) -> Optional[Tuple[int, ...]]:
        """Atomically reserve ``num_pages`` pages for a request.

        Returns the request's block table or ``None`` on insufficient capacity;
        a failed allocation never leaves a partial reservation.
        """
        self._check_id(request_id)
        if request_id in self._block_tables:
            raise BlockPoolError(f"request {request_id!r} already holds pages")
        if num_pages < 1:
            raise BlockPoolError(f"num_pages must be >= 1, got {num_pages}")
        if num_pages > len(self._free):
            return None
        pages = sorted(self._free)[:num_pages]
        self._block_tables[request_id] = []
        self._active[request_id] = set()
        self._reserve(request_id, pages)
        return tuple(pages)

    def extend(self, request_id: str, num_pages: int) -> Optional[Tuple[int, ...]]:
        """Reserve ``num_pages`` more pages for a growing sequence (atomic)."""
        self._check_id(request_id)
        if request_id not in self._block_tables:
            raise BlockPoolError(f"request {request_id!r} holds no pages")
        if num_pages < 1:
            raise BlockPoolError(f"num_pages must be >= 1, got {num_pages}")
        if num_pages > len(self._free):
            return None
        pages = sorted(self._free)[:num_pages]
        self._reserve(request_id, pages)
        return tuple(pages)

    def block_table(self, request_id: str) -> Tuple[int, ...]:
        """Immutable view of a request's assigned pages."""
        self._check_id(request_id)
        if request_id not in self._block_tables:
            raise BlockPoolError(f"request {request_id!r} holds no pages")
        return tuple(self._block_tables[request_id])

    def mark_in_flight(self, request_id: str) -> None:
        """Hold an execution reference on every active page of a request.

        One execution reference per request at a time: a repeated call while the
        request is already in flight is a no-op rather than a second reference.
        """
        self._check_id(request_id)
        if request_id not in self._active:
            raise BlockPoolError(f"request {request_id!r} holds no active pages")
        if request_id in self._in_flight:
            return
        self._in_flight[request_id] = set(self._active[request_id])
        for page in self._in_flight[request_id]:
            self._refs[page] += 1

    def complete_in_flight(self, request_id: str) -> None:
        """Drop the execution reference; pages free when no refs remain."""
        self._check_id(request_id)
        held = self._in_flight.pop(request_id, None)
        if held is not None:
            self._drop_refs(held)

    def release(self, request_id: str) -> None:
        """Drop ownership on a terminal path; in-flight pages stay reserved."""
        self._reclaim(request_id)

    def cancel(self, request_id: str) -> None:
        """Cancel from any nonterminal state; in-flight pages stay reserved."""
        self._reclaim(request_id)

    def _reclaim(self, request_id: str) -> None:
        self._check_id(request_id)
        active = self._active.pop(request_id, None)
        if active is None:
            return
        self._block_tables.pop(request_id, None)
        self._drop_refs(active)

    def owns(self, request_id: str) -> bool:
        return request_id in self._block_tables

    @property
    def num_free_pages(self) -> int:
        return len(self._free)

    @property
    def num_allocated_pages(self) -> int:
        return self.num_pages - len(self._free)

    @property
    def is_exhausted(self) -> bool:
        return not self._free