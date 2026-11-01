"""The model runner every family shares, around tinygrad's GGUF transformer.

Tinygrad owns loading, transformer math, recurrence, caches, RNG and JIT. The engine adds bounded
generation and its lifecycle, and on NV swaps quantized projections for infurnace's packed kernels,
attention and the dense feed-forward for fused ones, and reads prompts in fixed-size chunks. What differs
between model families (tokenizer, extra passes such as Qwen3.5's fused GDN, prompt snapshots) comes from
their Family in infurnace/models.
"""
import array
from dataclasses import dataclass
import functools
import hashlib
import math
import os
from pathlib import Path
from typing import Iterator
from tinygrad import Context, Tensor, TinyJit, UOp, dtypes
from tinygrad.device import Buffer
from tinygrad.llm.model import Transformer, TransformerBlock
from tinygrad.nn.state import get_state_dict
from infurnace.weights.gguf import Q2_0, lazy_upstream_weights, parse_gguf, read_gguf_metadata
from infurnace.weights.cache import cache_root, file_identity, load_tensors, prune_cache, save_tensors, source_hash
from infurnace.quant.hadamard import rotations
from infurnace.quant.linear import PackedEmbedding, PackedLinear, _NVLinear
from infurnace.layers.fused import FusibleNorm, fused_attention, fused_feed_forward, int8_kv_cache
from infurnace.kernels.quant_gemm import nv_prefill_supported
from infurnace.device import nvidia
from infurnace.models import Family, family_for

@dataclass(frozen=True)
class ModelConfig:
  """Runner-facing hyperparameters from tinygrad's loaded transformer."""
  block_count: int
  embedding_length: int
  feed_forward_length: int
  attention_head_count: int
  attention_head_count_kv: int
  key_length: int
  value_length: int
  rope_dimension_count: int
  rope_freq_base: float
  rms_norm_epsilon: float
  vocab_size: int
  context_length: int

# Prompt chunks: fixed-size graphs (see Model._prompt_chunk); GDN layers scan them 16 rows at a time. Bigger chunks
# read each weight for more rows: Bonsai-4B (RTX 5060) reads prompts at 5055 tok/s in 64 rows, 5612 in 128, 5642 in
# 256 (2k; 8k: 5290, 5578, 5714; 16k: 4959, 5306, 5468) and 4261 in 512 (16k). A chunk costs its full size however
# few rows are prompt (a 256-row one 85 ms, a 64-row one 32), so the rest of a prompt goes SHORT_ROWS at a time once
# a big chunk would cost more than the short ones it replaces.
PROMPT_ROWS = 256
SHORT_ROWS = 64
# Capturing the prompt graph costs ~20-30 s and a one-row step ~9 ms (RTX 5060), so outside warmup it is
# captured only for a prompt long enough to pay for it.
PREFILL_CAPTURE_TOKENS = 2048
# Decoding queues this many steps before reading their tokens back (after bursts of 1, 2 and 4): each step reads
# its input from the device and writes its sample there, so the GPU doesn't wait on the host between them. Bonsai-4B
# plain decoding, RTX 5060 (2k / 16k context): 167 / 145 tok/s reading every token, 174 / 154 with bursts of 4,
# 183 / 158 with 8. A stop wastes up to 7 steps (~40 ms).
DECODE_BURST = 8

SNAPSHOTS = 8  # prompt snapshots kept (recurrent families), about 50 MB of host memory each for Qwen3.5-4B

@dataclass
class _Snapshot:
  prefix: list[int]
  owner: object                 # the DFlash draft whose state was saved too, or None
  draft_state: object
  states: Tensor                # recurrent and conv states, flat: on the device when saved, the CPU after offload_snapshots
  user: bool                    # saved at the end of a user message

class Model:
  """Bounded single-request runner around tinygrad's transformer, for any family in infurnace/models."""
  def __init__(self, transformer: Transformer, metadata: dict, device: str | None = None, family: 'Family | None' = None):
    if not transformer.blk: raise ValueError('the model must contain transformer blocks')
    c = transformer.blk[0].config
    self.family = family or family_for(metadata.get('general.architecture'))
    self.config = ModelConfig(c.num_blocks, c.dim, c.hidden_dim, c.n_heads, c.n_kv_heads,
      c.head_dim, c.v_head_dim, c.rope_dim, c.rope_theta, c.norm_eps, c.vocab_size, transformer.max_context)
    self.transformer, self.metadata = transformer, metadata
    self.device = device or transformer.token_embd.weight.device
    self._generating = False
    self.nv_quant_linears = 0
    self.nv_prefill = False
    self._path, self._host_embedding = None, None
    self._tied_head = False  # the packed output head is the token embedding table
    self._embedding: PackedEmbedding | None = None  # its lookup, when it is (see lookup)
    self._multirow_prefill = False  # prompts are read in prompt chunks (see _prefill)
    self._warming = False
    # Prompt chunks: a captured step for each size that slices the prompt from a persistent device buffer (see _prompt_chunk)
    self._prompt_jits = {size: TinyJit(functools.partial(self._prompt_chunk, size=size)) for size in (PROMPT_ROWS, SHORT_ROWS)}
    # A prompt snapshot's copy of the recurrent states into its own buffer: one captured step (see save_snapshot)
    self._snapshot_jit = TinyJit(self._snapshot_copy)
    self._prompt_ids: Tensor | None = None
    # Decoding steps that read their token from that buffer and write their sample after it (see _decode)
    self._decode_jit = TinyJit(self._decode_step)
    # Target layers whose inputs a prompt chunk also returns (DFlash's feature taps; set by its generator)
    self.feature_layers: tuple[int, ...] = ()
    # Saved prompt states for later turns (see save_snapshot). turn_token marks where turns start;
    # user_turn (its header's tokens) and tool_marker tell a user message from a tool result.
    self.turn_token: int | None = None
    self.user_turn: tuple[int, ...] = ()
    self.tool_marker: int | None = None
    self._snapshots: list[_Snapshot] = []

  def warmup(self, prompt_graph: bool = True) -> None:
    """Prime upstream prefill/decode JIT using the same chunk size as normal generation. Without
    `prompt_graph`, the prompt-chunk graph is left for capture_prompt_graph (~11 s of the warmup)."""
    if self.config.context_length < 2: return
    # Long enough to capture the prompt graphs too (a big chunk, a short one and a partial short one).
    prompt = [0] * min(PROMPT_ROWS + SHORT_ROWS + 2 if prompt_graph else 8, self.config.context_length - 1)
    count = min(4, self.config.context_length - len(prompt))
    self._warming, multirow = True, self._multirow_prefill
    if not prompt_graph: self._multirow_prefill = False
    try:
      # First calls execute, second calls capture, third calls verify replay.
      for _ in range(3):
        stream = self.generate(prompt, count, reuse_cache=False)
        try:
          for _ in stream: pass
        finally: stream.close()
    finally:
      self._warming, self._multirow_prefill = False, multirow
      self.reset()  # No dummy tokens become conversation history or reusable prefix.

  def capture_prompt_graph(self) -> bool:
    """Capture the prompt-chunk graphs not captured yet outside a request (a server's idle time, after a warmup
    whose prompt used only some), leaving the conversation as it was: each chunk is read at the end of the cache,
    past any prompt, and the recurrent states it advances are put back. True when it captured any."""
    t = self.transformer
    sizes = [size for size, jit in self._prompt_jits.items() if jit.captured is None]
    position = t.max_context - PROMPT_ROWS
    if not self._multirow_prefill or not sizes or self._generating or position < 0: return False
    self.realize_state()
    with Context(DEV=self.device):
      states = self._recurrent_states()
      saved = [state.clone().realize() for state in states]
      self.upload_prompt([0])
      for size in sizes:
        self._prompt_jits[size].cnt = 1  # capture right away (see _multirow_ready)
        self._read_chunk(size, position, 1)
      if states: Tensor.realize(*(Tensor(state.uop.after(state.uop.store(copy.uop))) for state, copy in zip(states, saved)))
    # Cache rows from `position` on now hold the capture's: forget a prefix or snapshot that reaches them.
    if len(t._cached_tokens) > position or any(len(s.prefix) > position for s in self._snapshots): self.reset()
    return True

  def realize_state(self) -> None:
    """Create upstream's lazily initialized state as real buffers: KV caches, RoPE tables, conv and
    recurrent states, and the device RNG (created without advancing its stream). Upstream otherwise
    creates them inside its first call, and a graph captured before that would re-create them,
    zeroed, on every replay."""
    with Context(DEV=self.device):
      x, states = Tensor.empty(1, 1, self.config.embedding_length), []
      for block in self.transformer.blk:
        block._init_state(x)
        states += [getattr(block, name) for name in ('cache_kv', 'cache_scale', 'freqs_cis', 'conv_state', 'recurrent_state')
                   if hasattr(block, name)]
      if self.device not in Tensor._device_seeds: Tensor._next_counter(self.device, 0)
      states += [Tensor._device_seeds[self.device], Tensor._device_rng_counters[self.device]]
      if pending := [t for t in states if not t.uop.is_realized]: Tensor.realize(*pending)

  def _prompt_chunk(self, ids: Tensor, start_pos: UOp, rows: UOp, size: int) -> Tensor:
    """Read prompt rows start_pos..start_pos+rows-1 (1..size) into the caches and states.

    The graph always runs `size` rows sliced from the whole prompt, already on the device; `rows` is only a
    value: GDN layers treat later rows as no-ops (β 0, no decay) and keep their conv window at `rows`, and the
    keys and values attention writes for them lie past the prompt, overwritten before anything reads them. A
    symbolic row count would give hundreds of kernels runtime launch sizes, which the GPU patches on every
    replay: 65 ms per chunk instead of 33 (RTX 5060). Unlike upstream's forward, there is no output head or
    sample, and no host-side slice to schedule and upload."""
    t, features = self.transformer, []
    for block in t.blk: block._prompt_rows = rows  # recurrent layers mask the rest (see models/qwen35.py)
    try:
      x, mask = t.token_embd(ids[:, start_pos:start_pos + size]).float(), None
      for index, block in enumerate(t.blk):
        if index in self.feature_layers: features.append(x)  # tap N is the input of block N
        x, mask = block(x, start_pos, mask)  # blocks return (output, the indexer's token mask)
    finally:
      for block in t.blk: block._prompt_rows = None
    # The last row's output depends on every layer's cache and state writes; the taps alone would leave out the
    # layers after the last tap, so they are returned beside it, never instead of it.
    last = x[:, -1:].contiguous()
    return (features[0].cat(*features[1:], dim=-1).contiguous(), last) if features else last

  def set_feature_layers(self, layers: tuple[int, ...]):
    """Have prompt chunks return the inputs of these target layers (DFlash's feature taps)."""
    if tuple(layers) != self.feature_layers:
      self.feature_layers = tuple(layers)
      for jit in self._prompt_jits.values(): jit.reset()  # graphs captured for other taps

  def upload_prompt(self, tokens: list[int]):
    """Put the whole prompt on the device once; each chunk's graph slices its rows from it. The buffer starts as
    zeros and only ever holds token ids after (prompts, decoding steps' samples), so a chunk's padding rows read
    valid ids: an upload writes the prompt's positions only, and none when they hold it already."""
    t = self.transformer
    with Context(DEV=self.device):
      if self._prompt_ids is None:
        self._prompt_ids = Tensor.zeros(1, t.max_context, dtype=dtypes.int32, device=self.device).contiguous().realize()
        self._uploaded: list[int] = []
      if self._uploaded == tokens: return
      # a plain copy into the buffer's first positions: a store kernel would compile anew for each prompt length
      host = memoryview(array.array('i', tokens)).cast('B')
      self._prompt_ids.uop.buffer.view(host.nbytes, 0).ensure_allocated().copy_from(Buffer('NPY', host.nbytes, opaque=host))
      self._uploaded = list(tokens)

  def read_prompt(self, first: int, last: int, consume=None) -> int:
    """Read uploaded prompt rows from `first` toward `last` in prompt chunks (PROMPT_ROWS while more than three
    short chunks remain, then SHORT_ROWS); returns where it stopped (short of `last` only near the end of the
    context: a chunk writes all its padded cache rows). With feature layers set, `consume(features, rows)` gets
    each chunk's taps (1, chunk size, width) and its row count before the next chunk overwrites them."""
    t, position = self.transformer, first
    while position < last:
      size = PROMPT_ROWS if last - position > 3 * SHORT_ROWS and position + PROMPT_ROWS <= t.max_context else SHORT_ROWS
      if position + size > t.max_context: break
      rows = min(size, last - position)
      out = self._read_chunk(size, position, rows)
      if consume is not None: consume(out[0], rows)
      position += rows
    return position

  def _read_chunk(self, size: int, position: int, rows: int):
    t = self.transformer
    v_start, v_rows = UOp.variable('start_pos', 0, t.max_context - 1), UOp.variable('prompt_rows', 1, size)
    with Context(DEV=self.device): return self._prompt_jits[size](self._prompt_ids, v_start.bind(position), v_rows.bind(rows))

  def reset(self) -> None:
    """Invalidate prefix reuse without reallocating tinygrad's cache or JIT."""
    if self._generating: raise RuntimeError("cannot reset during generation")
    self.transformer._cached_tokens = []
    self._snapshots = []

  # Prompt snapshots (recurrent families only), as llama.cpp's server keeps its context checkpoints. Recurrent state can't be
  # rewound, so a new prompt that shares only part of the last one would be read from scratch.
  # Instead the recurrent and conv states (and a DFlash draft's position) are copied to host memory
  # at one point of each prompt, just before its final turn header, and a later prompt resumes from
  # the newest one it still matches. Qwen's template re-renders earlier replies without their thinking
  # once a new user message arrives, so that message's prompt matches only up to the end of the
  # previous user message: the snapshot saved there is kept however long the tool loop after it ran.
  # The attention KV caches need no copy: they are indexed by position, and every snapshot is a prefix
  # of what the cache holds; one a new prompt diverges from is dropped when its positions are rewritten.

  def _recurrent_states(self) -> list[Tensor]:
    return [Tensor(getattr(block, name).uop.buf_uop).reshape(getattr(block, name).shape)
            for block in self.transformer.blk if hasattr(block, 'recurrent_state') for name in ('conv_state', 'recurrent_state')]

  def snapshot_point(self, tokens: list[int], start: int) -> int | None:
    """Where to save this prompt's state: before its last turn header, else 4 tokens from the end
    (a raw prompt). None when the state there is already gone (start is past it), or the family has no
    recurrent state: attention-only models resume from their KV cache alone."""
    if not self.family.recurrent: return None
    marks = [i for i, token in enumerate(tokens) if token == self.turn_token] if self.turn_token is not None else []
    point = marks[-1] if marks else len(tokens) - 4
    return point if 0 < point < len(tokens) and point >= start else None

  def _ends_user_message(self, prefix: list[int]) -> bool:
    """Whether the last message in `prefix` is a user message (not a tool result)."""
    if self.turn_token is None or not self.user_turn or self.turn_token not in prefix: return False
    last = len(prefix) - 1 - prefix[::-1].index(self.turn_token)
    header = tuple(prefix[last:last + len(self.user_turn)])
    return header == self.user_turn and prefix[last + len(header):last + len(header) + 1] != [self.tool_marker]

  def save_snapshot(self, prefix: list[int], draft=None) -> None:
    """Save the current state as the state after `prefix` (and the draft's, if it read the same rows)."""
    if any(snap.prefix == prefix and snap.owner is draft for snap in self._snapshots): return
    # One captured copy on the device into the snapshot's own buffer (a copy to the CPU per state cost ~3.5 ms of
    # host time each, and the transfer ~1 GB/s: ~170 ms of a Bonsai-4B prompt's first token; scheduling the 48
    # device copies ~40); offload_snapshots moves it to the CPU after the reply
    sizes = [state.numel() for state in self._recurrent_states()]
    with Context(DEV=self.device): states = self._snapshot_jit(Tensor.empty(sum(sizes), dtype=dtypes.float32).realize())
    snap = _Snapshot(list(prefix), draft, None if draft is None else draft.save_state(), states, self._ends_user_message(prefix))
    self._snapshots = sorted([s for s in self._snapshots if s.prefix != prefix] + [snap], key=lambda s: len(s.prefix))
    while len(self._snapshots) > SNAPSHOTS:  # the oldest goes, except the last user message's
      users = [s for s in self._snapshots if s.user]
      self._snapshots.remove(next(s for s in self._snapshots if not users or s is not users[-1]))

  def _snapshot_copy(self, dest: Tensor) -> Tensor:
    flat = Tensor.cat(*(state.float().flatten() for state in self._recurrent_states()))
    return Tensor(dest.uop.after(dest.uop.store(flat.uop))).realize()

  def offload_snapshots(self) -> None:
    """Move snapshots other than the newest to CPU memory (one schedule): run after a reply, off the time to the
    first token, so the device keeps one (~50 MB for Qwen3.5-4B)."""
    moving = [snap for snap in self._snapshots[:-1] if snap.states.device != 'CPU']
    if not moving: return
    with Context(DEV=self.device):
      copies = [snap.states.clone(device='CPU') for snap in moving]
      Tensor.realize(*copies)
    for snap, states in zip(moving, copies): snap.states = states

  def snapshot_length(self, tokens: list[int], draft=None) -> int:
    """How much of `tokens` the newest usable snapshot covers (0 without one)."""
    usable = [s for s in self._snapshots if len(tokens) > len(s.prefix) and tokens[:len(s.prefix)] == s.prefix
              and (draft is None or s.owner is draft)]
    return len(usable[-1].prefix) if usable else 0

  def restore_snapshot(self, tokens: list[int], start: int, draft=None) -> int:
    """The position to resume `tokens` from: the newest matching snapshot's, after restoring it, when
    that beats `start`. Snapshots this prompt diverges from past the resume point are dropped: their
    cache rows are about to be overwritten."""
    length = self.snapshot_length(tokens, draft)
    if length > start:
      snap = next(s for s in reversed(self._snapshots) if len(s.prefix) == length and (draft is None or s.owner is draft))
      with Context(DEV=self.device):
        flat, offset, writes = snap.states.to(self.device), 0, []
        for state in self._recurrent_states():
          part = flat[offset:offset + state.numel()].reshape(state.shape).cast(state.dtype)
          writes.append(Tensor(state.uop.after(state.uop.store(part.uop))))
          offset += state.numel()
        Tensor.realize(*writes)
      if draft is not None: draft.load_state(snap.draft_state)
      start = length
    self._snapshots = [s for s in self._snapshots if len(s.prefix) <= start or tokens[:len(s.prefix)] == s.prefix]
    return start

  def generate(self, tokens: list[int], max_new_tokens: int, temperature: float = 0.0, *,
               chunk_size: int = 32, reuse_cache: bool = True) -> Iterator[int]:
    """Bound upstream generation; callers check EOS and must close abandoned streams.

    Sampling and prefix reuse are upstream behavior. Context exhaustion bounds
    output; an already-full prompt is rejected, never silently truncated.
    """
    if self._generating: raise RuntimeError("only one generation may run at a time")
    if not isinstance(tokens, list) or not tokens: raise ValueError("prompt must contain at least one token")
    if any(type(t) is not int or not 0 <= t < self.config.vocab_size for t in tokens): raise ValueError("invalid prompt token ID")
    if type(max_new_tokens) is not int or max_new_tokens < 0: raise ValueError("max_new_tokens must be a nonnegative integer")
    if type(chunk_size) is not int or chunk_size <= 0: raise ValueError("chunk_size must be a positive integer")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature < 0:
      raise ValueError("temperature must be finite and nonnegative")
    if len(tokens) >= self.config.context_length: raise ValueError("prompt exceeds available context; shorten history or increase --max-context")
    if not reuse_cache: self.reset()
    count = min(max_new_tokens, self.config.context_length - len(tokens))
    if count == 0: return
    self._generating = True
    stream = None
    try:
      with Context(DEV=self.device): self._prefill(tokens, float(temperature))
      t = self.transformer
      if hasattr(self, 'sampler') and t.get_start_pos(tokens) == len(tokens) - 1: stream = self._decode(tokens, count, float(temperature))
      else: stream = t.generate(list(tokens), chunk_size=chunk_size, temperature=float(temperature))
      for _ in range(count):
        # Do not leave a device context active while yielding control to a caller.
        with Context(DEV=self.device): token = next(stream)
        yield token
    finally:
      if stream is not None: stream.close()
      self._generating = False
      self.offload_snapshots()

  def _decode_step(self, ids: Tensor, start_pos: UOp, temperature: Tensor) -> Tensor:
    """One decoding step: the token at start_pos of the device token buffer in, its sample stored at start_pos + 1."""
    out = self.transformer.forward(ids[:, start_pos:start_pos + 1], start_pos, temperature)
    return Tensor(ids.uop.after(ids[:, start_pos + 1:start_pos + 2].uop.store(out.cast(ids.dtype).uop)))

  def _decode(self, tokens: list[int], count: int, temperature: float) -> Iterator[int]:
    """Upstream's generate after _prefill (the last prompt token is the first step's input), but in bursts of
    steps queued back to back, each taking its input from the device token buffer, then one read of their tokens:
    the GPU no longer idles while the host reads each token and launches the next step (~0.7 ms a Bonsai-4B token,
    RTX 5060). A stop inside a burst leaves its later steps done: the cached prefix counts them, so the next prompt
    reuses the cache only if it extends them (a chat's next turn resumes from its prompt snapshot either way)."""
    t = self.transformer
    v_start = UOp.variable('start_pos', 0, t.max_context - 1)
    with Context(DEV=self.device):
      self.upload_prompt(tokens)
      temp = Tensor([temperature]).contiguous().realize()
    sequence, position, burst = list(tokens), len(tokens) - 1, 1
    while count > 0:
      steps = min(burst, count, t.max_context - 1 - position)
      if steps <= 0: return
      with Context(DEV=self.device):
        for i in range(steps): self._decode_jit(self._prompt_ids, v_start.bind(position + i), temp)
        new = self._prompt_ids[0, position + 1:position + 1 + steps].tolist()
      sequence += new
      position, count, burst = position + steps, count - steps, min(burst * 2, DECODE_BURST)
      t._cached_tokens = sequence[:position]  # the steps ran on every position before this one
      yield from new

  def _multirow_ready(self, rows: int) -> bool:
    """Whether to read `rows` prompt tokens in prompt chunks (capturing their graphs if needed)."""
    if not self._multirow_prefill or rows <= 0: return False
    if pending := [jit for jit in self._prompt_jits.values() if jit.captured is None]:
      # Not captured yet: worth it during warmup or for a long prompt, otherwise upstream reads
      # this prompt one row at a time (a short cold `--prompt` would wait ~19 s to save ~0.2 s).
      if not (self._warming or rows >= PREFILL_CAPTURE_TOKENS): return False
      # Skip TinyJit's eager first call and capture right away: with every state buffer already
      # real, the eager call has nothing left to set up.
      self.realize_state()
      for jit in pending: jit.cnt = 1
    return True

  def _prefill(self, tokens: list[int], temperature: float):
    """Read the prompt up to its last token, then record it as the cached prefix: upstream's generate
    resumes there and processes the last token as an ordinary one-row step.

    Resumes from the prompt snapshot when the prompt still matches it, and saves a new one on the way.
    Upstream prefills hybrid models one row per step except on AMD (its scan is a row-by-row loop
    elsewhere); when the family supports it, rows go through prompt chunks (_prompt_chunk)."""
    t = self.transformer
    end = len(tokens) - 1
    start = self.restore_snapshot(tokens, t.get_start_pos(tokens))
    point = self.snapshot_point(tokens, start)
    multirow = self._multirow_ready(end - start)
    v_start = UOp.variable('start_pos', 0, t.max_context - 1)
    ids = Tensor(tokens + [0] * (t.max_context - len(tokens)), dtype='int32').reshape(1, t.max_context)
    temp = Tensor([temperature])
    if multirow: self.upload_prompt(tokens)
    def read(first: int, last: int):
      # Prompt chunks through the captured prompt step, then any rest (near the end of the context, or every
      # row without chunks) as one-row steps through upstream's rollout_jit with its own one-row `toks` variable.
      position = self.read_prompt(first, last) if multirow else first
      for position in range(position, last):
        sp, rows = v_start.bind(position), UOp.variable('toks', 1, 1).bind(1)
        t(ids[:, sp:sp + rows], sp, temp).realize()
    if point is not None:
      read(start, point)
      self.save_snapshot(tokens[:point])
      start = point
    if multirow and end > start:
      read(start, end)
      start = end
    t._cached_tokens = tokens[:start]

  def embed(self, tokens: list[int]) -> Tensor:
    """Embedding rows as a (1, len(tokens), dim) float tensor on the model device.

    Multirow one-hot gathers from the packed table scan the whole vocabulary on
    GPU. With a tied head, only the requested rows are decoded from its packed
    weights on device; otherwise a host FP16 copy turns lookups into an upload.
    Both round through FP16 like upstream's token embedding.
    """
    if self._path is None:
      from tinygrad import dtypes
      with Context(DEV=self.device):
        ids = Tensor([tokens], dtype=dtypes.int32, device=self.device)
        return Tensor.cat(*(self.transformer.token_embd(ids[:, r:r+1]) for r in range(len(tokens))), dim=1).float()
    from tinygrad import dtypes
    dim = self.config.embedding_length
    if self._embedding is not None:
      with Context(DEV=self.device):
        rows = self.lookup(Tensor(tokens, dtype=dtypes.int32, device=self.device))
        return rows.half().float().reshape(1, len(tokens), dim)
    if self._host_embedding is None:
      _, entries = parse_gguf(self._path)
      with Context(DEV='CPU'):
        table = entries['token_embd.weight'].to('CPU').decode().cast(dtypes.float16).contiguous().realize()
      self._host_embedding = (table, table.data().cast('B'))
    view, width = self._host_embedding[1], dim * 2
    rows = b''.join(view[t * width:(t + 1) * width] for t in tokens)
    return Tensor(rows, device=self.device).bitcast(dtypes.float16).reshape(1, len(tokens), dim).float()

  def lookup(self, ids: Tensor) -> Tensor:
    """FP32 token embedding rows for `ids` (int32 on the device, any shape), from the tied packed table: decoded and,
    for Hadamard-folded models, restored to the model's basis. Every embedding of tokens on the device goes through
    here or the token embedding it is, so a table's format or transform is handled once."""
    if self._embedding is None: raise ValueError('lookup needs a packed output head tied to the token embedding')
    return self._embedding.rows(ids)

  def _fuse_norms(self):
    """Hadamard-folded models: each block's norms become FusibleNorm where their consumer is infurnace's pass (which
    takes the norm's input for one row), so one-row decoding norms, rotates and quantizes in one kernel: ~65 fewer
    kernels a Bonsai-4B token, and the norm's own reduction and scaling kernels."""
    for block in self.transformer.blk:
      if getattr(getattr(block, 'attn_qkv', getattr(block, 'attn_q', None)), 'rotation', None) is None: continue
      if getattr(block._attention, 'takes_norm', False): block.attn_norm = FusibleNorm(block.attn_norm)
      if getattr(block._feed_forward, 'takes_norm', False): block.ffn_norm = FusibleNorm(block.ffn_norm)

  def _merge_projections(self):
    """Q2_0 projections of one input (attn_qkv and attn_gate, attn_q/k/v, ffn_gate and ffn_up) become one linear of
    their rows (_NVLinear.merged), run by _project: one launch instead of two or three. A small GEMV is short of
    the bandwidth a large one reaches (RTX 5060: ~280-330 GB/s at 6 MB, 400 at 179 MB), so fewer, larger ones help."""
    for block in self.transformer.blk:
      for names in (('attn_qkv', 'attn_gate'), ('attn_q', 'attn_k', 'attn_v'), ('ffn_gate', 'ffn_up')):
        layers = tuple(getattr(block, name, None) for name in names)
        if not all(isinstance(layer, _NVLinear) and layer.ggml_type == Q2_0 and layer.bias is None for layer in layers): continue
        if any(layer.in_features != layers[0].in_features or layer.rotation is not layers[0].rotation for layer in layers): continue
        layers[0].merged_group = (_NVLinear.merged(layers), layers)

  def _fuse_attention_prologue(self):
    """Blocks whose one-row attention can take kernels/attention.attention_prologue (gated q, per-head q/k norms, an
    int8 KV cache, merged q/k/v): their norm weights in FP32 once, which marks them."""
    for block in self.transformer.blk:
      c = block.config
      group = getattr(getattr(block, 'attn_q', None), 'merged_group', None)
      if group is None or not c.attn_output_gate or c.qk_norm != c.head_dim or c.rope_interleave or not 0 < c.rope_dim <= c.head_dim: continue
      if not nvidia(self.device) or c.head_dim % 32 or os.environ.get('INFURNACE_KV_INT8', '1') == '0': continue
      block.qk_norm_f32 = tuple(n.weight.float().contiguous().realize() for n in (block.attn_q_norm, block.attn_k_norm))

  def _sample_on_device(self):
    """Upstream's forward with its sample from kernels/sampling.py (the same Gumbel-max, noise made in the kernel)
    instead of tensor ops: upstream's random noise, two logs and two argmax passes over the 248,320 logits took
    ~0.53 ms of a ~6.6 ms Bonsai-4B decode step (RTX 5060). DFlash's target step samples from the same sampler, so
    its decisions keep matching plain decoding's."""
    from infurnace.kernels.sampling import Sampler
    t = self.transformer
    # Seeded from tinygrad's seed, so Tensor.manual_seed before loading makes sampled requests reproducible.
    self.sampler = Sampler(self.device, (Tensor._seed * 0xD1B54A32D192ED03 + 0x51A7E) & (2 ** 64 - 1))
    def forward(tokens: Tensor, start_pos, temperature: Tensor) -> Tensor:
      x, mask = t.token_embd(tokens.to(t.token_embd.weight.device)).float(), None
      for block in t.blk: x, mask = block(x, start_pos, mask)
      logits = t.output(t.output_norm(x[:, -1:]))[:, -1, :]
      return self.sampler(logits, temperature).reshape(1, 1).to(tokens.device)
    t.forward, t.prefill_jit, t.rollout_jit = forward, TinyJit(forward), TinyJit(forward)

  @classmethod
  def from_gguf(cls, path: str | Path, max_context: int | None = None, device: str = 'CPU', *,
                nv_quant: bool = False, nv_prefill: bool | None = None, family: 'Family | None' = None) -> 'Model':
    if max_context is not None and (type(max_context) is not int or max_context <= 0): raise ValueError('invalid max_context')
    if nv_prefill and not nv_quant: raise ValueError('NV prefill requires NV quantized acceleration')
    # Architecture dispatch is a cheap metadata-only check. Tensor loading,
    # configuration and block math remain upstream.
    family = family or family_for(read_gguf_metadata(path).get('general.architecture'))
    if not nv_quant:
      with Context(DEV=device):
        transformer, metadata = Transformer.from_gguf(Path(path), max_context=max_context, realize=False)
      return cls(transformer, metadata, family=family)
    if not nvidia(device): raise ValueError('packed projections require an NVIDIA device')
    # Stage upstream's raw file in host memory; only packed projections and the
    # small remaining parameters are copied to the GPU. Upstream otherwise
    # re-unpacks every quantized weight inside each step's kernels.
    with Context(DEV='CPU'), lazy_upstream_weights(path):
      transformer, metadata = Transformer.from_gguf(Path(path), max_context=max_context, realize=False)
    result = cls(transformer, metadata, device, family)
    result._path = Path(path)
    result._enable_packed(path, nv_prefill_supported(device) if nv_prefill is None else nv_prefill)
    if os.environ.get('INFURNACE_FUSED_ATTENTION', '1') != '0':
      for block in transformer.blk:
        if (type(block) is TransformerBlock and not block.config.sliding_window and not hasattr(block, 'attn_sinks')
            and block.config.head_dim % 32 == 0):
          block._attention = fused_attention(block)
          if nvidia(device) and block.config.head_dim % 128 == 0 and os.environ.get('INFURNACE_KV_INT8', '1') != '0':
            int8_kv_cache(block)
    for block in transformer.blk:
      if isinstance(getattr(block, 'ffn_gate', None), _NVLinear): block._feed_forward = fused_feed_forward(block)
    # The family's own passes; it says whether prompts can then be read in chunks.
    chunks = family.accelerate(result)
    if transformer.config.hc is None: result._sample_on_device()
    result._fuse_norms()
    result._merge_projections()
    result._fuse_attention_prologue()
    result._multirow_prefill = chunks and result.nv_prefill and os.environ.get('INFURNACE_MULTIROW_PREFILL', '1') != '0'
    # Placeholder initialization leaves an unrealized host RNG update chain that
    # every later realize would walk. Swap in a fresh counter rather than realize
    # it (that would materialize the placeholders); host sampling is unused.
    if (counter := Tensor._device_rng_counters.get('CPU')) is not None:
      counter.replace(Tensor.zeros(*counter.shape, dtype=counter.dtype, device='CPU').contiguous().realize())
    return result

  def _attach_rotations(self, path):
    """Hadamard-folded weights (hadamard.py): each projection rotates its input; the embedding restores its rows."""
    metadata, entries = parse_gguf(path)
    arch = metadata.get('general.architecture')
    widths = {name: entry.shape[-1] for name, entry in entries.items() if len(entry.shape) == 2}
    widths.setdefault('output.weight', widths.get('token_embd.weight', 0))
    forward, inverse = rotations(metadata, widths, self.device, value_heads=metadata.get(f'{arch}.ssm.time_step_rank', 0),
                                 key_heads=metadata.get(f'{arch}.ssm.group_count', 0))
    for name, rotation in forward.items():
      owner, attribute = (self.transformer, 'output') if name == 'output.weight' else \
                         (self.transformer.blk[int(name.split('.')[1])], name.split('.')[2])
      layer = getattr(owner, attribute, None)
      # A folded weight computed without its rotation would silently be a different model.
      if not isinstance(layer, _NVLinear): raise ValueError(f'Hadamard-folded weight {name} has no packed projection to rotate its input')
      layer.rotation = rotation
    if inverse is not None:
      if self._embedding is None or self.transformer.token_embd is not self._embedding:
        raise ValueError('a Hadamard-latent token embedding needs the packed lookup')
      self._embedding.rotation = inverse

  def _param_cache_path(self, path):
    if os.environ.get('INFURNACE_PARAM_CACHE', '1') == '0': return None
    key = hashlib.sha256(f'{source_hash()} {file_identity(path)} {self.device}'.encode()).hexdigest()[:32]
    return cache_root() / 'params' / f'{key}.bin'

  def _enable_packed(self, path, prefill: bool):
    self.nv_prefill = prefill
    _, entries = parse_gguf(path)
    # Converted Qwen3.5 GGUFs tie the output head to the token embedding.
    self._tied_head = 'output.weight' not in entries
    entries.setdefault('output.weight', entries['token_embd.weight'])
    packed, layers = set(), {}
    with Context(DEV=self.device):
      for name, entry in entries.items():
        if name == 'token_embd.weight' or len(entry.shape) != 2 or entry.shape[1] % 256 or entry.shape[0] % 16: continue
        if entry.ggml_type not in (12, 13, 14, Q2_0): continue
        if name == 'output.weight': owner, attribute = self.transformer, 'output'
        elif name.startswith('blk.'):
          _, layer, attribute, _ = name.split('.')
          owner = self.transformer.blk[int(layer)]
        else: continue
        tied = name == 'output.weight' and self._tied_head
        # Lazy copy + repack; each layer is realized in one schedule below
        # (one realize per weight costs ~20 ms of graph bookkeeping each).
        adapter = PackedLinear(getattr(owner, attribute), entry.to(self.device, realize=False), prefill=prefill,
                                keep_weight=tied, realize=False)
        layers.setdefault(name.split('.')[1] if name.startswith('blk.') else name, []).append(adapter.packed)
        setattr(owner, attribute, adapter)
        packed.add(name)
        if tied: self._embedding = PackedEmbedding(adapter)
        if tied and entry.ggml_type == Q2_0:
          # Upstream's embedding would decode the whole table: the packed lookup is the token embedding too, so no
          # dense decode graph stays alive (each realize walks it: ~5 ms of host time for Bonsai-4B's 248k-row head)
          self.transformer.token_embd = self._embedding
          adapter.weight = adapter.original.weight = None
        elif tied:
          # Embedding lookups read rows from the same packed allocation.
          self.transformer.token_embd.weight = adapter.weight
          adapter.weight = adapter.original.weight = None
        self.nv_quant_linears += 1
      self._attach_rotations(path)
      # Batches of ~512 MB of raw weights: few schedules, and the staged raw
      # copies of the whole file never coexist on the GPU.
      group, size = [], 0
      for packed_layer in layers.values():
        group += packed_layer
        size += sum(t.nbytes() for t in packed_layer)
        if size >= 512 << 20:
          Tensor.realize(*group)
          group, size = [], 0
      if group: Tensor.realize(*group)
      # Move what remains (norms, conv, decay, small Q8_0 gates) as realized FP16 tensors. Preparing
      # them schedules ~700 tiny kernels (~2 s) for ~10 MB, so the result is cached on disk.
      remaining = {name: tensor for name, tensor in get_state_dict(self.transformer).items() if tensor.device != self.device}
      cache = self._param_cache_path(path) if remaining else None
      cached = load_tensors(cache, self.device) if cache is not None and cache.is_file() else None
      if cached is not None and set(cached) == set(remaining) and all(
          cached[name].shape == tensor.shape and cached[name].dtype == tensor.dtype for name, tensor in remaining.items()):
        for name, tensor in remaining.items(): tensor.replace(cached[name])
      elif remaining:
        moved = [(tensor, tensor.to(self.device).contiguous()) for tensor in remaining.values()]
        Tensor.realize(*[copy for _, copy in moved])
        for tensor, copy in moved: tensor.replace(copy)
        if cache is not None:
          save_tensors(cache, remaining)
          prune_cache(cache.parent, keep=4)
