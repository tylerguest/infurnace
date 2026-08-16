from __future__ import annotations
import argparse
import os
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


def _build_real(artifact: str, manifest: str, *, num_slots: int,
                max_context: int | None, device: str | None) -> Engine:
    from infurnace.models.manifest import load_manifest
    from infurnace.executor.tinygrad.weights import load_qwen3_checkpoint
    from infurnace.executor.tinygrad.runner import Qwen3Runner
    from infurnace.tokenizer import GGUFTokenizer

    if not os.path.exists(artifact):
        raise SystemExit(f"artifact not found: {artifact} (download/place the GGUF locally)")
    if not os.path.exists(manifest):
        raise SystemExit(f"manifest not found: {manifest}")

    checkpoint = load_qwen3_checkpoint(artifact, load_manifest(manifest))
    tokenizer = GGUFTokenizer.from_gguf_metadata(checkpoint.metadata)
    runner = Qwen3Runner.from_weights(
        checkpoint.weights, num_slots=num_slots, max_context=max_context, device=device,
    )
    return Engine(runner, Scheduler(num_slots=num_slots), GreedySampler(), tokenizer)


def main() -> int:
    ap = argparse.ArgumentParser(description="infurnace offline text driver")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--stop-strings", nargs="*", default=[])
    ap.add_argument("--fake", action="store_true", help="CPU-only fake runner + tokenizer")
    ap.add_argument("--artifact", default="artifacts/models/Qwen3-0.6B-Q8_0.gguf")
    ap.add_argument("--manifest", default="models/qwen3-0.6b-q8_0.json")
    ap.add_argument("--num-slots", type=int, default=1)
    ap.add_argument("--max-context", type=int, default=2048,
                    help="KV cache context length (default 2048; lower saves GPU memory)")
    ap.add_argument("--device", default=None,
                    help="tinygrad device (sets TINYGRED); default: auto-detect (GPU if available)")
    args = ap.parse_args()

    if args.device:
        os.environ["TINYGRED"] = args.device

    eng = _build_fake() if args.fake else _build_real(
        args.artifact, args.manifest, num_slots=args.num_slots,
        max_context=args.max_context, device=args.device,
    )
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
