import torch
import numpy as np
import struct
import pdb
import ctypes
def bin2float(b):
  ''' Convert binary string to a float.
  Attributes:
  b: Binary string to transform.
  '''
  h = int(b, 2).to_bytes(8, byteorder="big")
  return struct.unpack('>d', h)[0]

def float2bin(f):
  ''' Convert float to 64-bit binary string.
  Attributes:
  :f: Float number to transform.
  '''
  [d] = struct.unpack(">Q", struct.pack(">d", f))
  return f'{d:032}'

def binary(num):
        return bin(ctypes.c_uint.from_buffer(ctypes.c_float(num)).value)
x = torch.randn(5, 3, 3)

x_numpy = x.numpy()
x_flatten = x_numpy.flatten()
pdb.set_trace()

print(binary(1.555))

