"""Opt-in EAGLE3 drafting and verification.

Modules: draft (checkpoint/model), generate (request orchestration),
reference (sequential verifier), verify (NV batched verifier), and sampling
(exact target decisions and RNG commitment). Ordinary decode stays in model.py.
"""
from infurnace.eagle.draft import EagleDraft

__all__ = ['EagleDraft']
