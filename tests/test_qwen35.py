import json
import struct
import pytest
from tinygrad import Context, Tensor
from tinygrad.llm.model import Transformer, TransformerConfig, SSMConfig, GatedDeltaNetBlock, TransformerBlock
from tinygrad.nn.state import get_state_dict
from infurnace.models.qwen35 import Qwen35Model


def checkpoint(path, *, permuted_head=False):
  """Small native GGUF fixture using upstream block/tensor names, no HF adapter."""
  c = TransformerConfig(num_blocks=2, dim=8, hidden_dim=12, n_heads=2, n_kv_heads=1, norm_eps=1e-6, vocab_size=16,
    head_dim=4, v_head_dim=4, max_context=16, rope_dim=2, rope_theta=10000000, qk_norm=4, attn_output_gate=True,
    ssm_layers=(True, False), ssm=SSMConfig(3, 2, 2, 4, 8))
  transformer = Transformer(c)
  state = {}
  for name, tensor in get_state_dict(transformer).items():
    if name == 'output.weight': continue  # exercise upstream tied-head fallback
    value = Tensor.randn(*tensor.shape) * 0.05
    if name.endswith('norm.weight'): value = value + 1
    if name.endswith('ssm_a'): value = -value.abs() - 0.05
    name = name.replace('ffn_norm', 'post_attention_norm')
    state[name] = value
  if permuted_head:
    state['output.weight'] = state['token_embd.weight'][Tensor(list(range(1, 16)) + [0])]
  metadata = {'general.architecture': 'qwen35', 'qwen35.block_count': 2, 'qwen35.context_length': 16,
    'qwen35.embedding_length': 8, 'qwen35.feed_forward_length': 12, 'qwen35.attention.head_count': 2,
    'qwen35.attention.head_count_kv': 1, 'qwen35.attention.key_length': 4, 'qwen35.attention.value_length': 4,
    'qwen35.attention.layer_norm_rms_epsilon': 1e-6, 'qwen35.rope.dimension_count': 2, 'qwen35.rope.freq_base': 10000000.0,
    'qwen35.ssm.conv_kernel': 3, 'qwen35.ssm.state_size': 2, 'qwen35.ssm.group_count': 2,
    'qwen35.ssm.time_step_rank': 4, 'qwen35.ssm.inner_size': 8, 'qwen35.full_attention_interval': 2,
    'tokenizer.ggml.tokens': [str(i) for i in range(16)]}
  def string(value):
    encoded = value.encode()
    return struct.pack('<Q', len(encoded)) + encoded
  def field(value):
    if isinstance(value, str): return 8, string(value)
    if isinstance(value, int): return 4, struct.pack('<I', value)
    if isinstance(value, float): return 6, struct.pack('<f', value)
    return 9, struct.pack('<IQ', 8, len(value)) + b''.join(string(v) for v in value)
  header = b'GGUF' + struct.pack('<IQQ', 3, len(state), len(metadata))
  for key, value in metadata.items():
    kind, encoded = field(value)
    header += string(key) + struct.pack('<I', kind) + encoded
  payload = bytearray()
  for name, tensor in state.items():
    payload.extend(bytes((-len(payload)) % 32))
    offset = len(payload)
    values = tensor.flatten().tolist()
    payload.extend(struct.pack('<' + 'f'*len(values), *values))
    header += string(name) + struct.pack('<I', len(tensor.shape))
    header += struct.pack('<' + 'Q'*len(tensor.shape), *reversed(tensor.shape)) + struct.pack('<IQ', 0, offset)
  header += bytes((-len(header)) % 32)
  file = path / 'model.gguf'
  file.write_bytes(header + payload)
  return file


def test_upstream_loader_and_exact_block_types(tmp_path):
  with Context(DEV='CPU'):
    path = checkpoint(tmp_path)
    model = Qwen35Model.from_gguf(path, max_context=8, device='CPU')
    assert type(model.transformer) is Transformer
    assert type(model.transformer.blk[0]) is GatedDeltaNetBlock
    assert type(model.transformer.blk[1]) is TransformerBlock
    assert type(model.transformer.output).__module__ == 'tinygrad.llm.kernels.amd'
    assert model.config.context_length == 8 and model.config.rope_dimension_count == 2
    assert model.transformer.output.weight.uop.key == model.transformer.token_embd.weight.uop.key
    for cap in (0, -1, True):
      with pytest.raises(ValueError): Qwen35Model.from_gguf(path, max_context=cap)


def test_bounded_wrapper_matches_direct_upstream_rng_and_cache(tmp_path):
  with Context(DEV='CPU'):
    path = checkpoint(tmp_path)
    direct, metadata = Transformer.from_gguf(path, max_context=16, realize=False)
    model = Qwen35Model.from_gguf(path, device='CPU')
    assert model.metadata == metadata
    Tensor.manual_seed(123)
    stream = direct.generate([1, 2], temperature=0.0)
    try: expected = [next(stream) for _ in range(7)]
    finally: stream.close()
    following = Tensor.rand(8).tolist()
    Tensor.manual_seed(123)
    assert list(model.generate([1, 2], 7, reuse_cache=False)) == expected
    assert Tensor.rand(8).tolist() == following
    for _ in range(3): assert list(model.generate([1, 2], 7, reuse_cache=False)) == expected
    prompt = [1, 2] + expected + [4]
    extended = list(model.generate(prompt, 2))
    fresh = list(model.generate(prompt, 2, reuse_cache=False))
    assert extended == fresh
    model.reset()
    assert model.transformer._cached_tokens == []


@pytest.mark.parametrize('flag', ['nv_attention', 'nv_fused_attention', 'nv_prepared_weights'])
def test_cli_rejects_attention_overrides_before_loading(tmp_path, monkeypatch, flag):
  from infurnace import cli
  path = tmp_path / 'model.gguf'
  path.touch()
  monkeypatch.setattr('infurnace.gguf.read_gguf_metadata', lambda path: {'general.architecture': 'qwen35'})
  monkeypatch.setattr(Qwen35Model, 'from_gguf', lambda *a, **kw: pytest.fail('loaded target for forbidden override'))
  with pytest.raises(ValueError, match='upstream tinygrad owns attention'):
    cli._build(str(path), 16, 'CPU', **{flag: True})


@pytest.mark.parametrize('flag', ['nv_quant', 'nv_prefill'])
def test_cli_packed_projection_flags_require_nv_before_loading(tmp_path, monkeypatch, flag):
  from infurnace import cli
  path = tmp_path / 'model.gguf'
  path.touch()
  monkeypatch.setattr('infurnace.gguf.read_gguf_metadata', lambda path: {'general.architecture': 'qwen35'})
  monkeypatch.setattr(Qwen35Model, 'from_gguf', lambda *a, **kw: pytest.fail('loaded target without NV packed support'))
  with pytest.raises(ValueError, match='requires NV'):
    cli._build(str(path), 16, 'CPU', **{flag: True})


def test_tokenizer_reference(tmp_path):
  tokenizers = pytest.importorskip('tokenizers')
  from tokenizers import Regex, models, trainers, pre_tokenizers, decoders, normalizers
  from infurnace.tokenizer import Qwen35Tokenizer
  hf = tokenizers.Tokenizer(models.BPE())
  hf.normalizer = normalizers.NFC()
  pattern = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|"
             r" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
  hf.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.Split(Regex(pattern), behavior='isolated'), pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
  hf.decoder = decoders.ByteLevel()
  hf.train_from_iterator(['Hello 1 Café 中文 😀'], trainers.BpeTrainer(
    vocab_size=300, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    special_tokens=['<|endoftext|>', '<|im_start|>', '<|im_end|>']))
  hf.save(str(tmp_path / 'tokenizer.json'))
  data = json.loads((tmp_path / 'tokenizer.json').read_text())
  words = sorted(data['model']['vocab'], key=data['model']['vocab'].get)
  special = {entry['id'] for entry in data['added_tokens']}
  tokenizer = Qwen35Tokenizer.from_gguf({'tokenizer.ggml.model': 'gpt2', 'tokenizer.ggml.pre': 'qwen35',
    'tokenizer.ggml.tokens': words, 'tokenizer.ggml.token_type': [3 if i in special else 1 for i in range(len(words))],
    'tokenizer.ggml.merges': [' '.join(pair) for pair in data['model']['merges']],
    'tokenizer.ggml.eos_token_id': data['model']['vocab']['<|im_end|>']})
  for text in ('Hello 1', 'Cafe\u0301 中文 😀', '<|im_start|>user\nHello<|im_end|>'):
    ids = tokenizer.encode(text)
    assert ids == hf.encode(text, add_special_tokens=False).ids
    decoded = hf.decode(ids, skip_special_tokens=False)
    assert tokenizer.decode(ids) == decoded
    stream = tokenizer.stream_decoder()
    assert ''.join(stream(i) for i in ids) + stream(None) == decoded
  assert tokenizer.is_end(tokenizer.eos_id) and tokenizer.is_end(tokenizer.eot_id)
