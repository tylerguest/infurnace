from __future__ import annotations
import argparse
import sys

from infurnace.engine import Engine
from infurnace.engine.request import SamplingParams
from infurnace.scheduler.scheduler import Scheduler
from infurnace.sampler import GreedySampler
from infurnace.tokenizer import Tokenizer


class FakeTokenizer(Tokenizer):
    """Trivial char<->token mapping for CPU-only offline runs."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)

    def is_end(self, token_id: int) -> bool:
        return False


def _build_fake() -> Engine:
    try:
        from tests.fakes import FakeRunner
    except Exception:
        from fakes import FakeRunner
    runner = FakeRunner(vocab_size=300, num_slots=1)
    return Engine(runner, Scheduler(num_slots=1), GreedySampler(), FakeTokenizer())


def _build_real(artifact: str) -> Engine:
    # Real Qwen3Runner + GGUFTokenizer wiring lands in Phase 3E.
    raise NotImplementedError("real runner integration is Phase 3E")


def main() -> int:
    ap = argparse.ArgumentParser(description="infurnace offline text driver")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--stop-strings", nargs="*", default=[])
    ap.add_argument("--fake", action="store_true", help="CPU-only fake runner + tokenizer")
    ap.add_argument("--artifact", default=None, help="GGUF artifact (real backend, Phase 3E)")
    args = ap.parse_args()

    eng = _build_fake() if args.fake else _build_real(args.artifact)
    eng.add_text_request(
        args.prompt,
        SamplingParams(max_tokens=args.max_tokens),
        stop_strings=args.stop_strings,
    )
    out = sys.stdout
    while not eng.is_done():
        for text in eng.step().new_text.values():
            if text:
                out.write(text)
                out.flush()
    out.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
