"""Owned NV host-mapped scratch for small EAGLE inputs and outputs.

No allocator/copy monkeypatches. CPU writes synchronize prior GPU users; tinygrad
synchronizes reads through Tensor.item/tolist. Owners retain output tensors for
as long as their JIT captures can replay.
"""
from array import array
import math

from tinygrad import Device, Tensor, UOp, dtypes
from tinygrad.device import Buffer, BufferSpec


def mapped_tensor(shape, dtype, device):
  if device.split(':')[0] != 'NV': raise ValueError('mapped EAGLE scratch requires NV')
  storage = Buffer(device, math.prod(shape), dtype, options=BufferSpec(host=True, cpu_access=True), preallocate=True)
  return Tensor(UOp.from_buffer(storage, device)).reshape(shape)


def upload_tokens(tensor, values, *, mapped=False):
  words = array('i', values)
  if words.itemsize != 4: raise RuntimeError('token upload requires 32-bit native integers')
  destination = tensor.uop.buffer.ensure_allocated()
  source = memoryview(words).cast('B')
  if tensor.dtype != dtypes.int32 or destination.nbytes != len(source): raise ValueError('invalid token scratch shape/dtype')
  Device[tensor.device].synchronize()
  if mapped:
    if tensor.device.split(':')[0] != 'NV' or destination.get_storage().host is None:
      raise ValueError('token destination is not NV host-mapped scratch')
    destination.host[:] = source
  else:
    destination.copy_from(Buffer('PYTHON', len(values), dtypes.int32, opaque=source))


def mapped_output(outputs, key, value):
  if key not in outputs: outputs[key] = mapped_tensor(value.shape, value.dtype, value.device)
  output = outputs[key]
  if output.shape != value.shape or output.dtype != value.dtype or output.device != value.device:
    raise ValueError('incompatible mapped output reuse')
  return Tensor(output.uop.buf_uop).reshape(value.shape).assign(value).realize()
