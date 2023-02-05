# Adapted from: https://github.com/phoboslab/qoi & https://github.com/mathpn/py-qoi 2022

"""
MIT License

Copyright (c) 2022 Dominic Szablewski

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from collections import defaultdict
from tabulate import tabulate
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import json
import zlib
import sys

from codec.curve import GeneralizedHilbertCurve
from codec.cluster import Partitioner
from codec.packbits import PackBits

# Header byte tags
class Utils:

	TAG_MASK = 0xc0       # 11000000
	DATA_MASK = ~TAG_MASK # 00111111

	TAG_DELTA = 0x00     # 0xxxxxxx
	DELTA_MASK = 0x80    # 1xxxxxxx

	# 01xxxxxx (0x40) NOT ALLOWED

	TAG_12_FULL = 0x80     # 10xxxxxx
	TAG_RUN = 0xc0       # 11xxxxxx
	
	# TAG_FULL_OLD = 0xfe  # 11111110
	TAG_FULL = 0xff      # 11111111

def unsign(x, n_bits):
	max_value = 2 ** n_bits # same as 1 << n_bits
	return (x + max_value) % max_value

def signed(x, n_bits):
	max_value = 2 ** n_bits
	if x > max_value / 2:
		x -= max_value
	return x

class Cache:

	def __init__(self):
		self.data = []
	
	def contains(self, value):
		return value in self.data
	
	def add(self, value):
		if self.contains(value):
			value = self.data.pop(self.data.index(value))
		self.data.insert(0, value)
		if len(self.data) > 64:
			self.data.pop()

	def get_index(self, value):
		if self.contains(value):
			return self.data.index(value)
		return None

class Pixel:

	def __init__(self, n = 2):
		self.n = n
		self.px_bytes = bytearray((0,) * n)
		self.value = 0

	def update(self, values: bytes) -> None:
		self.px_bytes[0 : self.n] = values
		self.value = int.from_bytes(self.px_bytes, byteorder = sys.byteorder)

	def __str__(self) -> str:
		return f'{tuple(self.px_bytes)}'

	def __eq__(self, other):
		return self.px_bytes == other.px_bytes

	def __hash__(self):
		return self.value

	@property
	def bytes(self) -> bytes:
		return bytes(self.px_bytes)

	@property
	def wrap_hash(self) -> int:
		return self.value % 64

class ByteWriter:

	def __init__(self):
		self.bytes = bytearray()

	def set_byte(self, index, value):
		self.bytes[index] = value

	def get_byte(self, index):
		return self.bytes[index]

	def write(self, byte: int):
		self.bytes.append(byte % 256)

	def write_4_bytes(self, value):
		self.write((0xff000000 & value) >> 24)
		self.write((0x00ff0000 & value) >> 16)
		self.write((0x0000ff00 & value) >> 8)
		self.write((0x000000ff & value))

	def write_2_bytes(self, value):
		self.write((0x0000ff00 & value) >> 8)
		self.write((0x000000ff & value))

	def output(self):
		return bytes(self.bytes)


class ByteReader:

	# FIXME: Adapts based on configured EOF
	padding_len = 1

	def __init__(self, data: bytes):
		self.bytes = data
		self.read_pos = 0
		self.max_pos = len(self.bytes) - self.padding_len - 1

	def read(self):

		if self.read_pos > self.max_pos:
			return None

		out = self.bytes[self.read_pos]
		self.read_pos += 1
		return out

	def read_4_bytes(self):
		data = [self.read() for _ in range(4)]
		b1, b2, b3, b4 = data
		return b1 << 24 | b2 << 16 | b3 << 8 | b4

	def read_2_bytes(self):
		data = [self.read() for _ in range(2)]
		b1, b2 = data
		return b1 << 8 | b2

	def output(self):
		return bytes(self.bytes[0 : self.read_pos])

class Encoder:

	def __init__(self, config, image, out_path = None):

		self.config = config

		self.image = image
		self.image_bytes = image.tobytes()

		# Split Byte[0] and Byte[1] for each 16 bit grayscale pixel value
		self.reorganized_bytes = self.image_bytes[0::2] + self.image_bytes[1::2]

		self.width, self.height = image.shape
		self.size = self.width * self.height

		self.writer = ByteWriter()
		self.out_path = out_path

		self.hash_array = [Pixel() for _ in range(64)]
		# self.cache = Cache()

		self.stats = [['Section', 'Size (KB)', 'Ratio (x)']]
		self.info = defaultdict(int)

	@property
	def MAGIC(self):
		a, b, c, d = self.config['magic']
		return ord(a) << 24 | ord(b) << 16 | ord(c) << 8 | ord(d)

	def encode_packbits(self):

		if self.config['verbose']:
			print(f'[PACKBITS CORE ENCODER FORMAT]')

		self.raw_size = self.size * self.config['channels'] * self.config['bytes_per_channel']

		self.stats.append(['Original', self.raw_size / 1000, 1.0])

		if self.raw_size > 400_000_000_000:
			raise MemoryError(f"Maximum byte count exceeded: {self.raw_size}")

		# Write header
		self.writer.write_4_bytes(self.MAGIC)

		# Write dimensions
		self.writer.write_2_bytes(self.width)
		self.writer.write_2_bytes(self.height)

		# Write color bit depth
		self.writer.write(self.config['channels'])
		self.writer.write(self.config['bytes_per_channel'])

		# Fractal configuration
		self.writer.write(int(self.config['fractal_transform']))

		if self.config['fractal_transform']:
			self.curve = GeneralizedHilbertCurve(self.width, self.height, get_index = True)
			pixel_order = self.curve.generator()
		else:
			pixel_order = range(self.size)

		self.packbits = PackBits(self.config['delta_transform'])
		packbits_encoded = self.packbits.encode(self.reorganized_bytes)
		for data in packbits_encoded:
			self.writer.write(data)

		# Write EOF termination
		if self.config['end_of_file'] is not None:
			self.writer.write(self.config['end_of_file'])

		output = self.writer.output()

		ratio = self.raw_size / len(output)
		self.stats.append(['PackBits', len(output) / 1000, ratio])

		if self.config['deflate_compression']:

			compressed = zlib.compress(output, level = 9)
		
			zlib_ratio = len(output) / len(compressed)
			self.stats.append(['DEFLATE', len(compressed) / 1000, zlib_ratio])

			output = compressed

		ratio = self.raw_size / len(output)
		self.stats.append(['Final', len(output) / 1000, ratio])

		if self.config['verbose']:
			table = tabulate(self.stats, headers = 'firstrow', tablefmt = 'simple_outline')
			print(table)

		if self.out_path is not None:
			with open(self.out_path, 'wb') as fout:
				fout.write(output)

		return output
	
	def encode_qoi(self):

		if self.config['verbose']:
			print(f'[QOI CORE ENCODER FORMAT]')

		self.raw_size = self.size * self.config['channels'] * self.config['bytes_per_channel']
		if self.raw_size > 400_000_000_000:
			raise MemoryError(f"Maximum byte count exceeded: {self.raw_size}")

		if not self.config['delta_transform']:
			raise NotImplementedError("Non-delta encoding not supported")

		self.stats.append(['Original', self.raw_size / 1000, 1.0])
		
		# Write header
		self.writer.write_4_bytes(self.MAGIC)

		# Image dimensions
		self.writer.write_2_bytes(self.width)
		self.writer.write_2_bytes(self.height)

		# Color bit depth format
		self.writer.write(self.config['channels'])
		self.writer.write(self.config['bytes_per_channel'])

		# Fractal configuration
		self.writer.write(int(self.config['fractal_transform']))

		pixel_jump = self.config['channels'] * self.config['bytes_per_channel']

		run = 0

		prev_pixel = Pixel()
		curr_pixel = Pixel()

		n = -1

		self.curve = GeneralizedHilbertCurve(self.width, self.height, get_index = True)
		if self.config['fractal_transform']:
			# pixel_order = self.curve.generator()
			pixel_order = self.curve.generate_all()
		else:
			pixel_order = range(self.size)

		if self.config['zipper_transform']:
			pixel_order = self.curve.zipper_transform(pixel_order)

		if self.config['segmentation_transform']:
			pixels = self.image.tolist()
			from itertools import chain
			pixels = list(chain.from_iterable(pixels))
			data = [pixels[i] for i in pixel_order]
			self.partition = Partitioner(data, block_size = 8)
			self.partition.initial_cluster()
			groups = self.partition.iterative_joining()
			self.cluster_jumps = self.partition.get_group_jumps()
			# print(self.cluster_jumps)
			if self.config['verbose']:
				print(len(self.cluster_jumps))

			pixel_order = groups

		for i in pixel_order:

			if self.config['segmentation_transform']:
				px = i.to_bytes(2, sys.byteorder)
			else:
				index = pixel_jump * i
				px = self.image_bytes[index : index + pixel_jump]

			n += 1

			prev_pixel.update(curr_pixel.bytes)
			curr_pixel.update(px)

			# Run length encoding
			if curr_pixel == prev_pixel:
				run += 1
				if run == 62 or (n + 1) >= self.size:
					self.writer.write(Utils.TAG_RUN | (run - 1))
					run = 0
				continue

			if run:
				self.info['run'] += 1
				self.writer.write(Utils.TAG_RUN | (run - 1))
				run = 0

			# Encoding values
			# index_pos = curr_pixel.wrap_hash
			delta = curr_pixel.value - prev_pixel.value

			# Near delta encoding
			if -64 < delta < 65:
				self.info['delta'] += 1
				self.writer.write(Utils.TAG_DELTA | unsign(delta, 7))
				# self.hash_array[index_pos].update(curr_pixel.bytes)
				continue

			# if self.hash_array[index_pos] == curr_pixel:
			# 	self.info['cache'] += 1
			# 	self.writer.write(Utils.TAG_CACHE | index_pos)
			# 	continue
			# self.hash_array[index_pos].update(curr_pixel.bytes)

			### QUERY CLUSTER JUMPING

			"""

			if n in self.cluster_jumps:
				completed.add(n)
				jump = self.cluster_jumps[n]
				n += jump

				self.writer.write(Utils.TAG_JUMP | unsign(jump, 6))

				continue
			
			"""

			# Full pixel delta encoding
			self.info['full'] += 1
			# self.writer.write(Utils.TAG_FULL)
			self.writer.write_2_bytes((Utils.TAG_12_FULL << 8) | unsign(delta, 12))

		if self.config['verbose']:
			print('\n' + json.dumps(self.info))

		# Write EOF termination
		if self.config['end_of_file'] is not None:
			self.writer.write(self.config['end_of_file'])

		output = self.writer.output()

		ratio = len(self.image_bytes) / len(output)
		self.stats.append(['Initial', len(output) / 1000, ratio])

		if self.config['deflate_compression']:

			compressed = zlib.compress(output, level = 9)

			zlib_ratio = len(output) / len(compressed)
			self.stats.append(['DEFLATE', len(compressed) / 1000, zlib_ratio])

			output = compressed

		ratio = len(self.image_bytes) / len(output)
		self.stats.append(['Final', len(output) / 1000, ratio])

		if self.config['verbose']:
			table = tabulate(self.stats, headers = 'firstrow', tablefmt = 'simple_outline')
			print(table)
		
		if self.out_path is not None:
			with open(self.out_path, 'wb') as fout:
				fout.write(output)

		return output

class Decoder:

	def __init__(self, config, file_bytes, out_path = None):

		self.config = config

		self.file_bytes = file_bytes
		if self.config['deflate_compression']:
			self.file_bytes = zlib.decompress(self.file_bytes)

		self.reader = ByteReader(self.file_bytes)
		self.out_path = out_path

		self.hash_array = [Pixel() for _ in range(64)]

	@property
	def MAGIC(self):
		a, b, c, d = self.config['magic']
		return ord(a) << 24 | ord(b) << 16 | ord(c) << 8 | ord(d)

	def decode_qoi(self):

		header_magic = self.reader.read_4_bytes()
		if header_magic != self.MAGIC:
			raise ValueError('Image does not contain valid header')

		self.width = self.reader.read_2_bytes()
		self.height = self.reader.read_2_bytes()

		self.channels = self.reader.read()
		self.bytes_per_channel = self.reader.read()

		self.fractal_transform = bool(self.reader.read())

		pixel_jump = self.channels * self.bytes_per_channel

		self.shape = (self.width, self.height)
		self.size = self.width * self.height
		self.total_size = self.width * self.height * pixel_jump

		self.pixel_data = bytearray(self.total_size)

		run = 0
		pixel = Pixel()
		prev_value = 0

		n = -1
		index = -1

		if self.fractal_transform:
			self.curve = GeneralizedHilbertCurve(self.width, self.height, get_index = True)
			pixel_order = self.curve.generator()
		else:
			pixel_order = range(self.size)

		self.fulls = []

		for i in pixel_order:

			n += 1

			index_pos = pixel.wrap_hash
			self.hash_array[index_pos].update(pixel.bytes)

			if index >= 0:
				self.pixel_data[index : index + pixel_jump] = pixel.bytes
				prev_value = pixel.value

			index = pixel_jump * i

			if run > 0:
				# self.fulls.append(i)
				run -= 1
				continue

			data = self.reader.read()
			if data is None:
				break
		
			# if data == Utils.TAG_FULL:
			# 	delta = signed(self.reader.read_2_bytes(), 16)
			# 	recovered = (prev_value + delta)
			# 	pixel.update(recovered.to_bytes(2, sys.byteorder))
			# 	self.fulls.append(i)
			# 	continue

			if (data & Utils.TAG_MASK) == Utils.TAG_12_FULL:
				
				rest = self.reader.read()
				full_data = data << 8 | rest

				delta = signed(full_data & 0xFFF, 12)

				recovered = (prev_value + delta)
				pixel.update(recovered.to_bytes(2, sys.byteorder))
				self.fulls.append(i)

				continue

			if (data & Utils.DELTA_MASK) == Utils.TAG_DELTA:
				delta = signed(~Utils.DELTA_MASK & data, 7) # signed(data, 7)
				recovered = (prev_value + delta)
				pixel.update(recovered.to_bytes(2, sys.byteorder))
				# self.fulls.append(i)
				continue

			# if (data & Utils.TAG_MASK) == Utils.TAG_CACHE:
			# 	pixel.update(self.hash_array[Utils.DATA_MASK & data].bytes)
			# 	# self.fulls.append(i)
			# 	continue

			if (data & Utils.TAG_MASK) == Utils.TAG_RUN:
				run = (data & 0x3f)
				# self.fulls.append(i)

		if self.out_path is not None:

			pixels = np.frombuffer(bytes(self.pixel_data), dtype = 'uint16').reshape(self.width, self.height)

			# pixels = np.zeros(self.size, dtype = np.uint16)
			# pixels[self.fulls] = 4095
			# pixels = pixels.reshape(self.width, self.height)

			preview = pixels.copy()

			# preview[:, ::8] = 4095
			# preview[::8, :] = 4095

			# Scale from 12 bit image to 16 bit display
			preview *= np.uint16(65536 / 4096)

			import imageio
			imageio.imwrite(self.out_path, preview)

			return pixels

		return self.pixel_data
	
	
	def decode_packbits(self):

		header_magic = self.reader.read_4_bytes()

		if header_magic != self.MAGIC:
			raise ValueError('Image does not contain valid header')

		self.width = self.reader.read_2_bytes()
		self.height = self.reader.read_2_bytes()

		self.channels = self.reader.read()
		self.bytes_per_channel = self.reader.read()

		self.fractal_transform = bool(self.reader.read())

		self.shape = (self.width, self.height)
		self.size = self.width * self.height
		self.total_size = self.width * self.height * self.channels * self.bytes_per_channel

		self.pixel_data = bytearray(self.total_size)

		rest = self.reader.bytes[self.reader.read_pos : -1]
		self.packbits = PackBits(self.config['delta_transform'])
		decoded = self.packbits.decode(rest)

		# Interleaving previously split bytes
		half = int(len(decoded) / self.bytes_per_channel)
		inter = [val for pair in zip(decoded[:half], decoded[half:]) for val in pair]
		
		if self.out_path is not None:

			arr = np.frombuffer(bytes(inter), dtype = 'uint16').reshape(self.width, self.height)
			
			plt.imshow(arr, cmap = 'gray')
			plt.savefig(self.out_path, bbox_inches = 'tight')
		
			return arr
