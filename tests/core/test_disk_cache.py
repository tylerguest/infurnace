from tinygrad import Context, Tensor, dtypes
from infurnace import disk_cache


def test_tensors_round_trip_as_views_of_one_buffer(tmp_path):
  with Context(DEV='CPU'):
    tensors = {'a': Tensor([1.5, -2.0, 3.25]).half().contiguous().realize(),
               'b.weight': Tensor.arange(12).reshape(3, 4).cast(dtypes.int32).contiguous().realize(),
               'c': Tensor([7.0]).contiguous().realize()}
    path = tmp_path / 'params.bin'
    disk_cache.save_tensors(path, tensors)
    loaded = disk_cache.load_tensors(path, 'CPU')
    assert set(loaded) == set(tensors)
    for name, tensor in tensors.items():
      assert (loaded[name].shape, loaded[name].dtype) == (tensor.shape, tensor.dtype)
      assert loaded[name].tolist() == tensor.tolist()
    bases = {loaded[name].uop.base.buffer._base for name in loaded}
    assert len(bases) == 1 and None not in bases  # one copy in, every tensor a view


def test_damaged_tensor_file_is_a_miss(tmp_path):
  path = tmp_path / 'params.bin'
  path.write_bytes(b'nonsense')
  assert disk_cache.load_tensors(path, 'CPU') is None
  assert disk_cache.load_tensors(tmp_path / 'missing.bin', 'CPU') is None
  with Context(DEV='CPU'): disk_cache.save_tensors(path, {'a': Tensor([1.0, 2.0]).contiguous().realize()})
  path.write_bytes(path.read_bytes()[:-256])  # data block cut off
  assert disk_cache.load_tensors(path, 'CPU') is None
