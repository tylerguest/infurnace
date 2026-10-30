"""Qwen3.5: hybrid Gated DeltaNet (GDN) and attention layers.

Its family adds the fused GDN pass on NV and prompt snapshots (GDN state can't be rewound); the tokenizer is
BPETokenizer. Everything else is the shared engine."""
import functools
import os
from tinygrad import Context, Tensor, UOp
from tinygrad.llm.model import GatedDeltaNetBlock
from infurnace.engine import PREFILL_ROWS, _norm_of, _norm_q8, _norm_rows_q8, _one_row, _project, _rows
from infurnace.kernels.gdn import gdn_gates, gdn_decode, gdn_prefill
from infurnace.models import Family
from infurnace.tokenizer import BPETokenizer

def _fused_gdn(block: GatedDeltaNetBlock):
  """Upstream GatedDeltaNetBlock._attention with NV kernels where upstream has them only on AMD.

  One-row decoding: the conv, q/k norms, recurrence, RMSNorm and gate in one kernel (elsewhere the
  step is dozens of small kernels). Prompt chunks (multiples of 16 rows): upstream's input processing, then
  closed-form kernels for the scan, 16 rows each, instead of its row-by-row loop. Other shapes use upstream's path."""
  upstream_attention = block._attention
  H, K, V = block.num_v_heads, block.head_k_dim, block.head_v_dim
  # β and α projections stacked for one gdn_gates launch (prompt chunks, and DFlash's verifier reads it from the block)
  gates = block.gdn_gates = block.ssm_beta.weight.cat(block.ssm_alpha.weight).contiguous().realize()
  # the decode kernel's FP32 conv and norm weights, made once (cast in each captured step, they were a kernel each)
  conv_w, norm_w = (w.float().contiguous().realize() for w in (block.ssm_conv1d['weight'], block.ssm_norm.weight))
  def prefill(x: Tensor, start_pos: UOp) -> Tensor:
    # Upstream's input processing, verbatim for the non-KDA case; only the scan below differs.
    B, T, _ = x.shape
    T_pad = x.max_shape[1]
    initial = Tensor(start_pos).eq(0)
    norm = _norm_of(block, 'attn_norm')
    source, rotation = norm.take(x) if norm is not None else None, block.attn_qkv.rotation
    if source is not None and block.attn_qkv.rows_quantized and block.attn_gate.rotation is rotation:
      # the norm, rotation and quantization in one kernel, which also hands back the FP16 normed rows the pass reads
      q8, x = _norm_rows_q8(block.attn_qkv, norm, source, half=True, normed=True)
      qkv, out_gate = _project((block.attn_qkv, block.attn_gate), x, q8, one_row=False)
    else:
      x = x.half()
      qkv, out_gate = _rows(x, block.attn_qkv, block.attn_gate)
    out_gate = out_gate.reshape(B, T, H, V)
    if isinstance(T, int):
      projected = gdn_gates(x.reshape(B * T, -1), gates).reshape(B, T, 2 * H)
      beta_in, alpha_in = projected[..., :H], projected[..., H:]
    else: beta_in, alpha_in = block.ssm_beta(x), block.ssm_alpha(x)
    beta = beta_in.sigmoid().reshape(B, T, H)
    log_alpha = (alpha_in.float() + block.ssm_dt['bias']).softplus().reshape(B, T, H, -1) * block.ssm_a.reshape(H, -1)
    conv_state = initial.where(0, block.conv_state)
    conv_window = conv_state.cat(qkv.cast(conv_state.dtype), dim=1)
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
    # The scan, RMSNorm and gate in one launch that also advances the recurrent state in place (16 rows a step),
    # ordered after the conv state write like upstream's AMD kernel.
    log_decay = log_alpha.transpose(1, 2)[0, ..., 0]                                   # (H, T_pad)
    state = Tensor(block.recurrent_state.uop.after(conv_state_store)).reshape(H, V, K)
    z = gdn_prefill(q[0], k[0], v[0], beta[0], log_decay, state, out_gate[0], block.ssm_norm.weight,
                    (1 - initial.float()).reshape(1), block.ssm_norm.eps)
    return block.ssm_out(z.reshape(1, T_pad, H * V)[:, :T])
  def attention(x: Tensor, start_pos, idx_mask=None):
    # idx_mask: an indexer's token selection (DeepSeek-style sparse attention), which these models don't have
    if idx_mask is not None: return upstream_attention(x, start_pos, idx_mask)
    B, T, _ = x.shape
    if not isinstance(start_pos, UOp): start_pos = UOp.variable('start_pos', 0, block.config.max_context - 1).bind(start_pos)
    multirow = isinstance(T, UOp) or T > 1  # prompt chunks have a symbolic row count
    if B == 1 and multirow and x.max_shape[1] % PREFILL_ROWS == 0: return prefill(x, start_pos)
    if B != 1 or multirow: return upstream_attention(x, start_pos)
    initial = Tensor(start_pos).eq(0)
    norm = _norm_of(block, 'attn_norm')
    source, rotation = norm.take(x) if norm is not None else None, block.attn_qkv.rotation
    if source is not None and rotation is not None and block.attn_gate.rotation is rotation:
      # the norm, rotation and quantization in one kernel, which also hands back the FP16 normed row the pass reads
      q8, x = _norm_q8(norm, source, rotation, half=True, normed=True)
      # merged: the kernel reads the qkv channels and the gate from the one output (no copies of its halves)
      outs = _project((block.attn_qkv, block.attn_gate), x, q8, one_row=True, joined=True)
      qkv, out_gate = (outs[0], None) if len(outs) == 1 else outs
    else:
      x = x.half()
      qkv, out_gate = _one_row(x, block.attn_qkv, block.attn_gate)
    if out_gate is not None: out_gate = out_gate.reshape(H, V)
    # beta/decay projections, conv, q/k norms, recurrence, RMSNorm and gate in one kernel; then the conv shift
    qkv = qkv.cast(block.conv_state.dtype).reshape(-1)
    conv_state = Tensor(block.conv_state.uop.buf_uop).reshape(block.ssm_conv_kernel - 1, block.conv_channels)
    state = Tensor(block.recurrent_state.uop.buf_uop).reshape(H, V, K)
    z = gdn_decode(qkv, conv_state, conv_w, x.reshape(-1), block.ssm_beta.weight, block.ssm_alpha.weight,
                   block.ssm_dt['bias'], block.ssm_a, state, out_gate, norm_w, (1 - initial.float()).reshape(1),
                   block.ssm_norm.eps, kheads=block.num_k_heads)
    return block.ssm_out(z.reshape(1, 1, H * V))
  attention.takes_norm = True  # its one-row path takes a _FusibleNorm's input
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

FAMILY = Family('qwen35', ('qwen35',), BPETokenizer.from_gguf, accelerate, recurrent=True, turn_markers=turn_markers)
