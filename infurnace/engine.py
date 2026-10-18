"""The model runner every family shares, around tinygrad's GGUF transformer.

Tinygrad owns loading, transformer math, recurrence, caches, RNG and JIT. The engine adds bounded
generation and its lifecycle, and on NV swaps quantized projections for infurnace's packed kernels,
attention and the dense feed-forward for fused ones, and reads prompts in fixed-size chunks. What differs
between model families (tokenizer, extra passes such as Qwen3.5's fused GDN, prompt snapshots) comes from
their Family in infurnace/models.
"""
import contextlib
from dataclasses import dataclass
import functools
import hashlib
import json
import math
import os
import struct
from pathlib import Path
from typing import Iterator
from tinygrad import Context, Tensor, TinyJit, UOp, dtypes
from tinygrad.device import Buffer
import tinygrad.llm.model as upstream_model
from tinygrad.llm.model import Transformer, TransformerBlock, apply_rope
from tinygrad.nn.state import get_state_dict
from infurnace.gguf import Q2_0, decode as decode_packed, parse_gguf, read_gguf_metadata
from infurnace.hadamard import rotations
from infurnace.kernels.attention import attention
from infurnace.kernels.quant_gemm import nv_prefill_supported
from infurnace.kernels.quant_gemv import dequant_prepared_q2, q8_quantize
from infurnace.linear import GEMM_TYPES, _NVLinear
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

# The parameter cache lives under $XDG_CACHE_HOME/infurnace/params and is
# keyed on everything that changes what it holds. A key mismatch, missing file or read error is a miss.
def cache_root() -> Path:
  return Path(os.environ.get('XDG_CACHE_HOME', str(Path.home() / '.cache'))).expanduser() / 'infurnace'

@functools.cache
def source_hash() -> str:
  """Hash of the tinygrad and infurnace sources: any code change selects new entries."""
  import tinygrad, infurnace
  digest = hashlib.sha256()
  for module in (tinygrad, infurnace):
    package = Path(module.__file__).parent
    for path in sorted(package.rglob('*.py')):
      digest.update(str(path.relative_to(package)).encode())
      digest.update(path.read_bytes())
  return digest.hexdigest()

def file_identity(path: str | Path) -> str:
  stat = Path(path).stat()
  return f'{Path(path).resolve()} {stat.st_size} {stat.st_mtime_ns}'

def prune_cache(directory: Path, keep: int) -> None:
  # Finished entries only: a .tmp may be another process's write in progress.
  for old in sorted(directory.glob('*.bin'), key=lambda p: p.stat().st_mtime)[:-keep]: old.unlink(missing_ok=True)

_ALIGN, _MAGIC = 256, b'INFTENS1'

def _save_tensors(path: Path, tensors: dict[str, Tensor]) -> None:
  """Realized tensors as one blob, each 256-byte aligned, plus a name/dtype/shape/offset index."""
  index, chunks, offset = [], [], 0
  for name, tensor in tensors.items():
    data = bytes(tensor.data())
    index.append({'name': name, 'dtype': tensor.dtype.name, 'shape': list(tensor.shape), 'offset': offset, 'nbytes': len(data)})
    chunks.append(data + bytes(-len(data) % _ALIGN))
    offset += len(chunks[-1])
  header = json.dumps(index).encode()
  path.parent.mkdir(parents=True, exist_ok=True)
  partial = path.with_suffix('.tmp')
  with partial.open('wb') as f:
    f.write(_MAGIC + struct.pack('<Q', len(header)) + header)
    f.write(bytes(-f.tell() % _ALIGN))
    for chunk in chunks: f.write(chunk)
  partial.replace(path)

def _load_tensors(path: Path, device: str) -> dict[str, Tensor] | None:
  """One host read and one copy to `device`; each tensor is a view into that buffer."""
  try:
    raw = path.read_bytes()
    if raw[:8] != _MAGIC: return None
    size = struct.unpack_from('<Q', raw, 8)[0]
    index = json.loads(raw[16:16 + size])
    blob = raw[16 + size + (-(16 + size) % _ALIGN):]
    if any(entry['offset'] + entry['nbytes'] > len(blob) for entry in index): return None
    base = Buffer(device, max(len(blob), 1), initial_value=blob)
    return {entry['name']: Tensor(UOp.from_buffer(Buffer(device, entry['nbytes'], base=base, offset=entry['offset']).ensure_allocated(),
                                                  getattr(dtypes, entry['dtype']))).reshape(entry['shape']) for entry in index}
  except (OSError, ValueError, KeyError, TypeError, AttributeError, struct.error):
    return None

class _PackedLinear(_NVLinear):
  """Packed projection that keeps no dense decode graph alive.

  Every live lazy graph is walked on each realize, so retained per-weight
  dequantization graphs cost tens of milliseconds per call. Rows go through
  the packed GEMM when available, otherwise the multirow packed GEMV.
  """
  def __init__(self, original, entry, *, prefill: bool, keep_weight: bool = False, realize: bool = True):
    super().__init__(original, entry, prefill=prefill and entry.ggml_type in GEMM_TYPES, q6_aligned=True, realize=realize)
    if not keep_weight: self.weight = original.weight = None

  def __call__(self, x):
    if self.prefill: return super().__call__(x)
    return self.decode(x)

PREFILL_ROWS = 16  # prompt chunks are padded to a multiple of this (the fused GDN kernel scans 16 rows at a time)
# Prompt chunk: a fixed-size graph (see Model._prompt_chunk) reading 64 rows at once reads prompts ~2.5x
# faster than 16 (RTX 5060: ~1,600 vs ~650 tok/s). GDN layers scan it 16 rows at a time.
PROMPT_ROWS = 64
# Capturing the prompt graph costs ~20-30 s and a one-row step ~9 ms (RTX 5060), so outside warmup it is
# captured only for a prompt long enough to pay for it.
PREFILL_CAPTURE_TOKENS = 2048

_SHARED_Q8 = os.environ.get('INFURNACE_SHARED_Q8', '1') != '0'

def _one_row(x: Tensor, *layers) -> tuple[Tensor, ...]:
  """One-row outputs of projections that read the same input, quantizing it to Q8 once: each packed
  GEMV would otherwise quantize it again. Other inputs or layers go through their own call."""
  width = getattr(layers[0], 'in_features', None)
  if _SHARED_Q8 and all(isinstance(layer, _NVLinear) and layer.rotation is layers[0].rotation for layer in layers) and \
     isinstance(x.numel(), int) and x.numel() == width:
    if layers[0].rotation is not None: x = layers[0].rotation(x)  # Hadamard-folded weights: one rotation for all of them
    q8 = q8_quantize(x, 1, width)
    return tuple(layer.decode(x, q8) for layer in layers)
  return tuple(layer(x) for layer in layers)

def _fused_feed_forward(block):
  """Upstream's dense feed-forward, with gate and up sharing one Q8 activation for one row."""
  upstream_feed_forward = block._feed_forward
  def feed_forward(x: Tensor) -> Tensor:
    if hasattr(block, 'ffn_gate_exps') or not isinstance(x.numel(), int) or x.numel() != getattr(block.ffn_gate, 'in_features', None):
      return upstream_feed_forward(x)
    gate, up = _one_row(x, block.ffn_gate, block.ffn_up)
    return block.ffn_down(gate.silu().contiguous() * up)
  return feed_forward

def _fused_attention(block: TransformerBlock):
  """Upstream TransformerBlock._attention for one-row decoding and multirow prompt chunks, with the
  attention itself in kernels/attention.py: upstream's attention over a runtime length slows as the
  conversation grows (it swaps in its own kernel only on AMD, at this same spot). Other shapes use
  upstream's path."""
  upstream_attention, c = block._attention, block.config
  def attention_step(x: Tensor, start_pos):
    B, T, _ = x.shape
    if B == 1 and (isinstance(T, UOp) or T > 1) and x.max_shape[1] % PREFILL_ROWS == 0:
      # A prompt chunk (1..16 or 1..64 rows), padded to its maximum like the GDN prefill. The padding rows come
      # from zero inputs, so their cached keys and values are finite (the kernel reads every key below the
      # padded end), and the prompt's next rows overwrite them. The prompt reader keeps the padded end in the cache.
      x = x.pad_to((B, x.max_shape[1], x.shape[2]))
      return rows_attention(x, start_pos, block.attn_q(x), block.attn_k(x), block.attn_v(x))[:, :T]
    if B != 1 or isinstance(T, UOp) or T != 1: return upstream_attention(x, start_pos)
    return rows_attention(x, start_pos, *_one_row(x, block.attn_q, block.attn_k, block.attn_v))
  def rows_attention(x: Tensor, start_pos, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    B, T, _ = x.shape
    if c.qk_norm and c.qk_norm != c.head_dim: q, k = block.attn_q_norm(q), block.attn_k_norm(k)
    if c.attn_output_gate:
      qg = q.reshape(B, T, c.n_heads, 2, c.head_dim)
      q, gate = qg[:, :, :, 0, :], qg[:, :, :, 1, :].reshape(B, T, c.n_heads * c.head_dim)
    q = q.reshape(B, T, c.n_heads, c.head_dim).transpose(1, 2)
    k = k.reshape(B, T, c.n_kv_heads, c.head_dim).transpose(1, 2)
    v = v.reshape(B, T, c.n_kv_heads, c.head_dim).transpose(1, 2)
    if c.qk_norm == c.head_dim: q, k = block.attn_q_norm(q), block.attn_k_norm(k)
    q = apply_rope(q[..., :c.rope_dim], block.freqs_cis[start_pos:start_pos + T]).cat(q[..., c.rope_dim:], dim=-1)
    k = apply_rope(k[..., :c.rope_dim], block.freqs_cis[start_pos:start_pos + T]).cat(k[..., c.rope_dim:], dim=-1)
    store = block.cache_kv[:, :, :, start_pos:start_pos + T, :].uop.store(Tensor.stack(k, v).cast(block.cache_kv.dtype).uop)
    attn = attention(q, Tensor(block.cache_kv.uop.after(store)), start_pos)
    attn = attn.cast(q.dtype).transpose(1, 2).reshape(B, T, -1)
    return block.attn_output(attn * gate.sigmoid() if c.attn_output_gate else attn)
  return attention_step

@contextlib.contextmanager
def _lazy_upstream_weights(path):
  """Build upstream's Transformer without reading its weights into host memory.

  Upstream's gguf_shard realizes every tensor of the file (2.7 GB copied to host
  RAM, one realize each), but the packed load replaces nearly all of them with
  packed NV copies read straight from disk. Inside this block the weights stay
  lazy, so only the ones that are kept get read (when moved to the device), and
  upstream reuses the header we already parsed (same metadata and entries)."""
  parse, shard = upstream_model.gguf_parse, upstream_model.gguf_shard
  def cached_parse(fn):
    metadata, entries = parse_gguf(path)
    return metadata, {name: (entry.data, entry.shape, entry.ggml_type) for name, entry in entries.items()}
  def lazy_shard(entries, devices, shard_map=None):
    if shard_map or len(devices) != 1: return shard(entries, devices, shard_map)
    return {name: decode_packed(data.to(devices[0]).flatten(), math.prod(shape), typ).reshape(shape)
            for name, (data, shape, typ) in entries.items()}
  upstream_model.gguf_parse, upstream_model.gguf_shard = cached_parse, lazy_shard
  try: yield
  finally: upstream_model.gguf_parse, upstream_model.gguf_shard = parse, shard

class _PackedEmbedding:
  """Token embedding read from a packed Q2_0 table (prepare_q2's layout): only the looked-up rows are decoded,
  then restored from a Hadamard-latent basis when the table is rotated."""
  def __init__(self, table: _NVLinear):
    self.table, self.rotation = table, None  # the table's Hadamard rotation: rows are restored with its inverse

  @property
  def weight(self) -> Tensor:  # upstream's forward places the token ids on token_embd.weight's device
    return self.table.packed

  def __call__(self, ids: Tensor) -> Tensor:
    t, flat = self.table, ids.flatten()
    code_words = t.out_features * t.in_features // 16
    codes = t.packed[:code_words].reshape(t.out_features, t.in_features // 16)[flat]
    scales = t.packed[code_words:].reshape(t.out_features, t.in_features // 128)[flat]  # two FP16 scales a word
    x = dequant_prepared_q2(codes.flatten().cat(scales.flatten()), in_features=t.in_features).reshape(*ids.shape, t.in_features)
    return (self.rotation.inverse(x) if self.rotation is not None else x).half()

SNAPSHOTS = 8  # prompt snapshots kept (recurrent families), about 50 MB of host memory each for Qwen3.5-4B

@dataclass
class _Snapshot:
  prefix: list[int]
  owner: object                 # the DFlash draft whose state was saved too, or None
  draft_state: object
  states: list[Tensor]          # recurrent and conv states, on the CPU
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
    self._multirow_prefill = False  # prompts are read PROMPT_ROWS rows at a time (see _prefill)
    self._warming = False
    # Prompt chunks: a captured step that slices the prompt from a persistent device buffer (see _prompt_chunk)
    self._prompt_jit = TinyJit(self._prompt_chunk)
    self._prompt_ids: Tensor | None = None
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
    # Long enough to capture the multirow prompt graph too (one full chunk and a partial one).
    prompt = [0] * min(PROMPT_ROWS + 2 if prompt_graph else 8, self.config.context_length - 1)
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
    """Capture the prompt-chunk graph outside a request (a server's idle time), leaving the conversation as
    it was: the chunk is read at the end of the cache, past any prompt, and the recurrent states it advances
    are put back. True when it captured the graph."""
    t = self.transformer
    position = t.max_context - PROMPT_ROWS
    if not self._multirow_prefill or self._prompt_jit.captured is not None or self._generating or position < 0: return False
    self.realize_state()
    with Context(DEV=self.device):
      states = self._recurrent_states()
      saved = [state.clone().realize() for state in states]
      self.upload_prompt([0])
      self._prompt_jit.cnt = 1  # capture right away (see _multirow_ready)
      self.read_prompt(position, position + 1)
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
        states += [getattr(block, name) for name in ('cache_kv', 'freqs_cis', 'conv_state', 'recurrent_state') if hasattr(block, name)]
      if self.device not in Tensor._device_seeds: Tensor._next_counter(self.device, 0)
      states += [Tensor._device_seeds[self.device], Tensor._device_rng_counters[self.device]]
      if pending := [t for t in states if not t.uop.is_realized]: Tensor.realize(*pending)

  def _prompt_chunk(self, ids: Tensor, start_pos: UOp, rows: UOp) -> Tensor:
    """Read prompt rows start_pos..start_pos+rows-1 (1..PROMPT_ROWS) into the caches and states.

    The graph always runs PROMPT_ROWS rows sliced from the whole prompt, already on the device; `rows` is only a
    value: GDN layers treat later rows as no-ops (β 0, no decay) and keep their conv window at `rows`, and the
    keys and values attention writes for them lie past the prompt, overwritten before anything reads them. A
    symbolic row count would give hundreds of kernels runtime launch sizes, which the GPU patches on every
    replay: 65 ms per chunk instead of 33 (RTX 5060). Unlike upstream's forward, there is no output head or
    sample, and no host-side slice to schedule and upload."""
    t, features = self.transformer, []
    for block in t.blk: block._prompt_rows = rows  # recurrent layers mask the rest (see models/qwen35.py)
    try:
      x = t.token_embd(ids[:, start_pos:start_pos + PROMPT_ROWS]).float()
      for index, block in enumerate(t.blk):
        if index in self.feature_layers: features.append(x)  # tap N is the input of block N
        x = block(x, start_pos)
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
      self._prompt_jit.reset()  # a graph captured for other taps

  def upload_prompt(self, tokens: list[int]):
    """Put the whole prompt on the device once; each chunk's graph slices its rows from it."""
    t = self.transformer
    with Context(DEV=self.device):
      if self._prompt_ids is None: self._prompt_ids = Tensor.empty(1, t.max_context, dtype=dtypes.int32, device=self.device).realize()
      padded = Tensor(tokens + [0] * (t.max_context - len(tokens)), dtype=dtypes.int32).reshape(1, t.max_context)
      Tensor(self._prompt_ids.uop.after(self._prompt_ids.uop.store(padded.to(self.device).uop))).realize()

  def read_prompt(self, first: int, last: int, consume=None) -> int:
    """Read uploaded prompt rows from `first` toward `last` in PROMPT_ROWS chunks; returns where it stopped
    (short of `last` only near the end of the context: a chunk writes all its padded cache rows). With
    feature layers set, `consume(features, rows)` gets each chunk's taps (1, PROMPT_ROWS, width) and its row
    count before the next chunk overwrites them."""
    t, position = self.transformer, first
    v_start, v_rows = UOp.variable('start_pos', 0, t.max_context - 1), UOp.variable('prompt_rows', 1, PROMPT_ROWS)
    with Context(DEV=self.device):
      while position < last and position + PROMPT_ROWS <= t.max_context:
        rows = min(PROMPT_ROWS, last - position)
        out = self._prompt_jit(self._prompt_ids, v_start.bind(position), v_rows.bind(rows))
        if consume is not None: consume(out[0], rows)
        position += rows
    return position

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
    with Context(DEV=self.device): states = [state.clone(device='CPU').realize() for state in self._recurrent_states()]
    snap = _Snapshot(list(prefix), draft, None if draft is None else draft.save_state(), states, self._ends_user_message(prefix))
    self._snapshots = sorted([s for s in self._snapshots if s.prefix != prefix] + [snap], key=lambda s: len(s.prefix))
    while len(self._snapshots) > SNAPSHOTS:  # the oldest goes, except the last user message's
      users = [s for s in self._snapshots if s.user]
      self._snapshots.remove(next(s for s in self._snapshots if not users or s is not users[-1]))

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
        Tensor.realize(*(Tensor(state.uop.after(state.uop.store(copy.to(self.device).uop)))
                         for state, copy in zip(self._recurrent_states(), snap.states)))
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
      stream = self.transformer.generate(list(tokens), chunk_size=chunk_size, temperature=float(temperature))
      for _ in range(count):
        # Do not leave a device context active while yielding control to a caller.
        with Context(DEV=self.device): token = next(stream)
        yield token
    finally:
      if stream is not None: stream.close()
      self._generating = False

  def _multirow_ready(self, rows: int) -> bool:
    """Whether to read `rows` prompt tokens in PROMPT_ROWS chunks (capturing that graph if needed)."""
    if not self._multirow_prefill or rows <= 0: return False
    if self._prompt_jit.captured is None:
      # Not captured yet: worth it during warmup or for a long prompt, otherwise upstream reads
      # this prompt one row at a time (a short cold `--prompt` would wait ~19 s to save ~0.2 s).
      if not (self._warming or rows >= PREFILL_CAPTURE_TOKENS): return False
      # Skip TinyJit's eager first call and capture right away: with every state buffer already
      # real, the eager call has nothing left to set up.
      self.realize_state()
      self._prompt_jit.cnt = 1
    return True

  def _prefill(self, tokens: list[int], temperature: float):
    """Read the prompt up to its last token, then record it as the cached prefix: upstream's generate
    resumes there and processes the last token as an ordinary one-row step.

    Resumes from the prompt snapshot when the prompt still matches it, and saves a new one on the way.
    Upstream prefills hybrid models one row per step except on AMD (its scan is a row-by-row loop
    elsewhere); when the family supports it, rows go PROMPT_ROWS at a time through _prompt_chunk."""
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
    if self._tied_head:
      from infurnace.kernels.embedding import packed_gather
      head = self.transformer.output
      with Context(DEV=self.device):
        ids = Tensor(tokens, dtype=dtypes.int32, device=self.device)
        rows = packed_gather(ids, head.packed, width=head.in_features, ggml_type=head.ggml_type, q6_aligned=head.q6_aligned,
                             q6_tiled=head.q6_tiled, q4_tiled=head.q4_tiled)
        return rows.half().float().reshape(1, len(tokens), dim)
    if self._host_embedding is None:
      _, entries = parse_gguf(self._path)
      with Context(DEV='CPU'):
        table = entries['token_embd.weight'].to('CPU').decode().cast(dtypes.float16).contiguous().realize()
      self._host_embedding = (table, table.data().cast('B'))
    view, width = self._host_embedding[1], dim * 2
    rows = b''.join(view[t * width:(t + 1) * width] for t in tokens)
    return Tensor(rows, device=self.device).bitcast(dtypes.float16).reshape(1, len(tokens), dim).float()

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
    with Context(DEV='CPU'), _lazy_upstream_weights(path):
      transformer, metadata = Transformer.from_gguf(Path(path), max_context=max_context, realize=False)
    result = cls(transformer, metadata, device, family)
    result._path = Path(path)
    result._enable_packed(path, nv_prefill_supported(device) if nv_prefill is None else nv_prefill)
    if os.environ.get('INFURNACE_FUSED_ATTENTION', '1') != '0':
      for block in transformer.blk:
        if (type(block) is TransformerBlock and not block.config.sliding_window and not hasattr(block, 'attn_sinks')
            and block.config.head_dim % 32 == 0):
          block._attention = _fused_attention(block)
    for block in transformer.blk:
      if isinstance(getattr(block, 'ffn_gate', None), _NVLinear): block._feed_forward = _fused_feed_forward(block)
    # The family's own passes; it says whether prompts can then be read in chunks.
    chunks = family.accelerate(result)
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
      embedding = self.transformer.token_embd
      if not isinstance(embedding, _PackedEmbedding): raise ValueError('a Hadamard-latent token embedding needs the packed lookup')
      embedding.rotation = inverse

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
        adapter = _PackedLinear(getattr(owner, attribute), entry.to(self.device, realize=False), prefill=prefill,
                                keep_weight=tied, realize=False)
        layers.setdefault(name.split('.')[1] if name.startswith('blk.') else name, []).append(adapter.packed)
        setattr(owner, attribute, adapter)
        packed.add(name)
        if tied and entry.ggml_type not in GEMM_TYPES:
          # No row-gather kernel (Q2_0): lookups decode their rows from the packed table.
          self.transformer.token_embd = _PackedEmbedding(adapter)
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
      cached = _load_tensors(cache, self.device) if cache is not None and cache.is_file() else None
      if cached is not None and set(cached) == set(remaining) and all(
          cached[name].shape == tensor.shape and cached[name].dtype == tensor.dtype for name, tensor in remaining.items()):
        for name, tensor in remaining.items(): tensor.replace(cached[name])
      elif remaining:
        moved = [(tensor, tensor.to(self.device).contiguous()) for tensor in remaining.values()]
        Tensor.realize(*[copy for _, copy in moved])
        for tensor, copy in moved: tensor.replace(copy)
        if cache is not None:
          _save_tensors(cache, remaining)
          prune_cache(cache.parent, keep=4)
