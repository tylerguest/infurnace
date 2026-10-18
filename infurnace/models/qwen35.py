"""Qwen3.5: hybrid Gated DeltaNet (GDN) and attention layers.

Its family adds the fused GDN pass on NV and prompt snapshots (GDN state can't be rewound); the tokenizer is
Qwen35Tokenizer. Everything else is the shared engine."""
import functools
import os
from tinygrad import Context, Tensor, UOp
from tinygrad.llm.model import GatedDeltaNetBlock
from infurnace.engine import PREFILL_ROWS, _one_row
from infurnace.kernels.gdn import gdn_gates, gdn_decode, gdn_prefill
from infurnace.models import Family
from infurnace.tokenizer import Qwen35Tokenizer

def _fused_gdn(block: GatedDeltaNetBlock):
  """Upstream GatedDeltaNetBlock._attention with NV kernels where upstream has them only on AMD.

  One-row decoding: the conv, q/k norms, recurrence, RMSNorm and gate in one kernel (elsewhere the
  step is dozens of small kernels). Prompt chunks (multiples of 16 rows): upstream's input processing, then
  closed-form kernels for the scan, 16 rows each, instead of its row-by-row loop. Other shapes use upstream's path."""
  upstream_attention = block._attention
  H, K, V = block.num_v_heads, block.head_k_dim, block.head_v_dim
  # β and α projections stacked for one gdn_gates launch (prompt chunks, and DFlash's verifier reads it from the block)
  gates = block.gdn_gates = block.ssm_beta.weight.cat(block.ssm_alpha.weight).contiguous().realize()
  def prefill(x: Tensor, start_pos: UOp) -> Tensor:
    # Upstream's input processing, verbatim for the non-KDA case; only the scan below differs.
    B, T, _ = x.shape
    T_pad = x.max_shape[1]
    initial = Tensor(start_pos).eq(0)
    x = x.half()
    out_gate = block.attn_gate(x).reshape(B, T, H, V)
    if isinstance(T, int):
      projected = gdn_gates(x.reshape(B * T, -1), gates).reshape(B, T, 2 * H)
      beta_in, alpha_in = projected[..., :H], projected[..., H:]
    else: beta_in, alpha_in = block.ssm_beta(x), block.ssm_alpha(x)
    beta = beta_in.sigmoid().reshape(B, T, H)
    log_alpha = (alpha_in.float() + block.ssm_dt['bias']).softplus().reshape(B, T, H, -1) * block.ssm_a.reshape(H, -1)
    conv_state = initial.where(0, block.conv_state)
    conv_window = conv_state.cat(block.attn_qkv(x).cast(conv_state.dtype), dim=1)
    conv_window = conv_window.pad_to((B, block.ssm_conv_kernel - 1 + T_pad, block.conv_channels)).contiguous()
    # A prompt chunk always runs PROMPT_ROWS rows; only the first `live` are prompt (see Model._prompt_chunk).
    live = getattr(block, '_prompt_rows', None)
    end = T if live is None else live
    conv_state_store = block.conv_state.uop.store(conv_window[:, end:end + block.ssm_conv_kernel - 1].cast(block.conv_state.dtype).uop)
    conv_out = functools.reduce(lambda a, b: a + b,
      (conv_window[:, i:i + T_pad] * block.ssm_conv1d['weight'][:, i] for i in range(block.ssm_conv_kernel))).silu()
    # Padding rows are no-ops: beta 0 (no update) and log-decay 0 (no decay).
    out_gate, beta = out_gate.pad_to((B, T_pad, H, V)), beta.pad_to((B, T_pad, H))
    log_alpha = log_alpha.pad_to((B, T_pad, *log_alpha.shape[2:]))
    if live is not None:
      prompt = Tensor.arange(T_pad).reshape(1, T_pad, 1) < Tensor(live)
      beta, log_alpha = prompt.where(beta, 0), prompt.unsqueeze(-1).where(log_alpha, 0)
    q, k, v = conv_out.split([block.q_dim, block.q_dim, block.conv_channels - 2 * block.q_dim], dim=-1)
    q, k = (z.reshape(B, T_pad, block.num_k_heads, K).normalize(dim=-1, eps=1e-6).repeat(1, 1, H // block.num_k_heads, 1)
            for z in (q, k))
    v = v.reshape(B, T_pad, H, V)
    q, k, v, beta = (z.transpose(1, 2).float() for z in (q, k, v, beta))
    q = q * K**-0.5
    # The scan, RMSNorm and gate in one kernel that also advances the recurrent state in place,
    # ordered after the conv state write like upstream's AMD kernel.
    # A longer chunk is scanned 16 rows at a time, each scan continuing the state the previous one left.
    log_decay = log_alpha.transpose(1, 2)[0, ..., 0]                                   # (H, T_pad)
    state = Tensor(block.recurrent_state.uop.after(conv_state_store)).reshape(H, V, K)
    zs = []
    for first in range(0, T_pad, PREFILL_ROWS):
      rows = slice(first, first + PREFILL_ROWS)
      keep = (1 - initial.float()).reshape(1) if first == 0 else Tensor.ones(1)
      z = gdn_prefill(q[0, :, rows], k[0, :, rows], v[0, :, rows], beta[0, :, rows], log_decay[:, rows].cumsum(-1), state,
                      out_gate[0, rows], block.ssm_norm.weight, keep, block.ssm_norm.eps)
      zs.append(z)
      state = Tensor(state.uop.after(z.uop))
    z = zs[0].cat(*zs[1:]) if len(zs) > 1 else zs[0]
    return block.ssm_out(z.reshape(1, T_pad, H * V)[:, :T])
  def attention(x: Tensor, start_pos):
    B, T, _ = x.shape
    if not isinstance(start_pos, UOp): start_pos = UOp.variable('start_pos', 0, block.config.max_context - 1).bind(start_pos)
    multirow = isinstance(T, UOp) or T > 1  # prompt chunks have a symbolic row count
    if B == 1 and multirow and x.max_shape[1] % PREFILL_ROWS == 0: return prefill(x, start_pos)
    if B != 1 or multirow: return upstream_attention(x, start_pos)
    initial = Tensor(start_pos).eq(0)
    x = x.half()
    qkv, out_gate = _one_row(x, block.attn_qkv, block.attn_gate)
    out_gate = out_gate.reshape(H, V)
    # beta/decay projections, conv, q/k norms, recurrence, RMSNorm and gate in one kernel; then the conv shift
    qkv = qkv.cast(block.conv_state.dtype).reshape(-1)
    conv_state = Tensor(block.conv_state.uop.buf_uop).reshape(block.ssm_conv_kernel - 1, block.conv_channels)
    state = Tensor(block.recurrent_state.uop.buf_uop).reshape(H, V, K)
    z = gdn_decode(qkv, conv_state, block.ssm_conv1d['weight'], x.reshape(-1), block.ssm_beta.weight, block.ssm_alpha.weight,
                   block.ssm_dt['bias'], block.ssm_a, state, out_gate, block.ssm_norm.weight, (1 - initial.float()).reshape(1),
                   block.ssm_norm.eps, kheads=block.num_k_heads)
    return block.ssm_out(z.reshape(1, 1, H * V))
  return attention

def accelerate(model) -> bool:
  """The fused GDN pass; prompts can be read in chunks only when every GDN layer has it."""
  gdn = [block for block in model.transformer.blk if isinstance(block, GatedDeltaNetBlock)]
  fused = [block for block in gdn if getattr(block, 'ssm_g_a', None) is None and block.head_k_dim == 128 and block.head_v_dim == 128]
  if os.environ.get('INFURNACE_FUSED_GDN', '1') == '0': return False
  for block in fused:
    block._attention = _fused_gdn(block)
    # The fused step updates the recurrent state in place, so it must be a real buffer.
    with Context(DEV=model.device):
      block._init_state(Tensor.empty(1, 1, model.config.embedding_length))
      Tensor.realize(block.conv_state, block.recurrent_state)
  # Multirow prompts need every GDN layer on the fused kernel.
  return len(fused) == len(gdn)

def turn_markers(tokenizer) -> tuple[int | None, tuple[int, ...], int | None]:
  """Snapshots go just before the last turn header; a user turn's header tells it from a tool result."""
  if (backend := getattr(tokenizer, '_backend', None)) is None: return None, (), None
  special = backend._special_tokens
  turn = special.get("<|im_start|>")
  return turn, (turn, *tokenizer.encode("user\n")) if turn is not None else (), special.get("<tool_response>")

FAMILY = Family('qwen35', ('qwen35',), Qwen35Tokenizer.from_gguf, accelerate, recurrent=True, turn_markers=turn_markers)
