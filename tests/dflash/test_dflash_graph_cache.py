from tinygrad import Context, Tensor, TinyJit
from tinygrad.engine.jit import _TinyJit
from infurnace.dflash import graph_cache
from infurnace.dflash.parallel import _stored


def owned(weight):
  return {'weight': Tensor(weight).contiguous().realize(), 'out': Tensor.empty(3).realize()}


def test_cached_graph_rebinds_named_buffers_in_a_new_process_state(tmp_path):
  with Context(DEV='CPU'):
    first = owned([1.0, 2.0, 3.0])
    jit = TinyJit(lambda x: _stored(first['out'], x * first['weight']).realize())
    for _ in range(3): jit(Tensor([1.0, 1.0, 1.0]).contiguous().realize())
    assert first['out'].tolist() == [1.0, 2.0, 3.0]
    names = {name: t.uop.base.buffer for name, t in first.items()}
    path = tmp_path / 'graphs.pkl'
    graph_cache.save(path, {'scale': jit}, names)

    # Same names, different buffers and contents: the loaded graph must use these.
    second = owned([10.0, 20.0, 30.0])
    loaded = graph_cache.load(path, {name: t.uop.base.buffer for name, t in second.items()})
    replay = _TinyJit(None, loaded['scale'])
    replay(Tensor([2.0, 2.0, 2.0]).contiguous().realize())
    assert second['out'].tolist() == [20.0, 40.0, 60.0]
    assert first['out'].tolist() == [1.0, 2.0, 3.0]


def test_missing_named_buffer_is_a_cache_miss(tmp_path):
  with Context(DEV='CPU'):
    first = owned([1.0, 2.0, 3.0])
    jit = TinyJit(lambda x: _stored(first['out'], x * first['weight']).realize())
    for _ in range(3): jit(Tensor([1.0, 1.0, 1.0]).contiguous().realize())
    path = tmp_path / 'graphs.pkl'
    graph_cache.save(path, {'scale': jit}, {name: t.uop.base.buffer for name, t in first.items()})
    assert graph_cache.load(path, {'weight': owned([1.0, 1.0, 1.0])['weight'].uop.base.buffer}) is None
    path.write_bytes(b'not a pickle')
    assert graph_cache.load(path, {}) is None
