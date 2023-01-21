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

from collections import OrderedDict
from tabulate import tabulate
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import json
import zlib
import sys

from codec.curve import GeneralizedHilbertCurve
from codec.packbits import PackBits

# chunk tags
class Utils:

	TAG_MASK = 0xc0  # 11000000

	TAG_RUN = 0xc0  # 11xxxxxx
	TAG_DELTA = 0x40  # 01xxxxxx
	TAG_CACHE = 0x00  # 00xxxxxx
	# TAG_FULL = 0x80  # 10xxxxxx

	# TAG_FULL_OLD = 0xfe  # 11111110

	TAG_FULL = 0xff # 11111111

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
	padding_len = 8

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
		self.info = {
			'cache': 0,
			'delta': 0,
			'luma': 0,
			'run': 0,
			'full': 0
		}

	@property
	def MAGIC(self):
		a, b, c, d = self.config['magic']
		return ord(a) << 24 | ord(b) << 16 | ord(c) << 8 | ord(d)
	
	def encode_deflate(self):

		if self.config['verbose']:
			print(f'[DEFLATE CORE ENCODER FORMAT]')

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
		
		output = self.image_bytes

		if self.config['zlib_compression']:

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
		for end in self.config['end_of_file']:
			self.writer.write(end)

		output = self.writer.output()

		ratio = self.raw_size / len(output)
		self.stats.append(['PackBits', len(output) / 1000, ratio])

		if self.config['zlib_compression']:

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
		
		if self.config['fractal_transform']:
			self.curve = GeneralizedHilbertCurve(self.width, self.height, get_index = True)
			pixel_order = self.curve.generator()
		else:
			pixel_order = range(self.size)

		# for i in tqdm.tqdm(pixel_order, total = self.size):
		for i in pixel_order:

			n += 1

			index = pixel_jump * i
			px = self.image_bytes[index : index + pixel_jump]

			prev_pixel.update(curr_pixel.bytes)
			curr_pixel.update(px)

			# Run length encoding
			if curr_pixel == prev_pixel:
				run += 1
				if run == 62 or (i + 1) >= self.size:
					self.writer.write(Utils.TAG_RUN | (run - 1))
					run = 0
				continue

			if run:
				self.info['run'] += 1
				self.writer.write(Utils.TAG_RUN | (run - 1))
				run = 0

			# Encoding values
			index_pos = curr_pixel.wrap_hash
			delta = curr_pixel.value - prev_pixel.value

			# Near delta encoding
			if -32 < delta < 33:
				self.info['delta'] += 1
				self.writer.write(Utils.TAG_DELTA | unsign(delta, 6))
				self.hash_array[index_pos].update(curr_pixel.bytes)
				continue

			if self.hash_array[index_pos] == curr_pixel:
				self.info['cache'] += 1
				self.writer.write(Utils.TAG_CACHE | index_pos)
				continue
			self.hash_array[index_pos].update(curr_pixel.bytes)

			# Full delta encoding
			self.info['full'] += 1
			self.writer.write(Utils.TAG_FULL)
			self.writer.write_2_bytes(unsign(delta, 16))

		if self.config['verbose']:
			print('\n' + json.dumps(self.info))

		# Write EOF termination
		for end in self.config['end_of_file']:
			self.writer.write(end)

		output = self.writer.output()

		ratio = len(self.image_bytes) / len(output)
		self.stats.append(['Initial', len(output) / 1000, ratio])

		if self.config['zlib_compression']:

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
		if self.config['zlib_compression']:
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

		for i in pixel_order:

			n += 1

			index_pos = pixel.wrap_hash
			self.hash_array[index_pos].update(pixel.bytes)

			if index >= 0:
				self.pixel_data[index : index + pixel_jump] = pixel.bytes
				prev_value = pixel.value

			index = pixel_jump * i

			if run > 0:
				run -= 1
				continue

			data = self.reader.read()
			if data is None:
				break
		
			if data == Utils.TAG_FULL:
				delta = signed(self.reader.read_2_bytes(), 16)
				recovered = (prev_value + delta)
				pixel.update(recovered.to_bytes(2, sys.byteorder))
				continue

			if (data & Utils.TAG_MASK) == Utils.TAG_DELTA:
				delta = signed(~Utils.TAG_MASK & data, 6)
				recovered = (prev_value + delta)
				pixel.update(recovered.to_bytes(2, sys.byteorder))
				continue
			
			if (data & Utils.TAG_MASK) == Utils.TAG_CACHE:
				pixel.update(self.hash_array[data].bytes)
				continue

			if (data & Utils.TAG_MASK) == Utils.TAG_RUN:
				run = (data & 0x3f)

		if self.out_path is not None:

			arr = np.frombuffer(bytes(self.pixel_data), dtype = 'uint16').reshape(self.width, self.height)

			# image = Image.fromarray(arr)

			# p_image = Image.fromarray(arr)
			# p_image.save(self.out_path, format = self.config['decode_format'])

			data = plt.imshow(arr, cmap = 'gray', aspect = 'auto')
			plt.savefig(self.out_path)

			# image = Image.frombuffer('I;16', self.shape, bytes(inter), 'raw', 'I;16', 2 * self.width, 1)
			# image.save(self.out_path, format = self.config['decode_format'])

			return arr


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

		rest = self.reader.bytes[self.reader.read_pos : -len(self.config['end_of_file'])]
		self.packbits = PackBits(self.config['delta_transform'])
		decoded = self.packbits.decode(rest)

		# Interleaving previously split bytes
		half = int(len(decoded) / self.bytes_per_channel)
		inter = [val for pair in zip(decoded[:half], decoded[half:]) for val in pair]
		
		if self.out_path is not None:

			arr = np.frombuffer(bytes(inter), dtype = 'uint16').reshape(self.width, self.height)

			data = plt.imshow(arr, cmap = 'gray', aspect = 'auto')
			plt.savefig(self.out_path)
		
			return arr
