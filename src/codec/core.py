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
from codec.cluster import Partitioner, BlockPartitioner
from codec.packbits import PackBits

# Header byte tags
class Utils:
	"""
	https://github.com/phoboslab/qoi/commit/28954f7a9a82e4a23ec6926c26617c9657839971
	#define QOI_INDEX     0x00 // 00xxxxxx
	#define QOI_RUN       0x40 // 01xxxxxx
	#define QOI_DIFF_8    0x80 // 10xxxxxx
	#define QOI_DIFF_16   0xc0 // 1100xxxx
	#define QOI_GDIFF_16  0xd0 // 1101xxxx
	#define QOI_DIFF_24   0xe0 // 1110xxxx
	#define QOI_COLOR     0xf0 // 1111xxxx
	"""

	"""
	https://github.com/phoboslab/qoi/commit/19dc63cf17456433f2d0086e5fa0f6372018bdc7
	#define QOI_INDEX   0x00 // 00xxxxxx
	#define QOI_RUN_8   0x40 // 010xxxxx
	#define QOI_RUN_16  0x60 // 011xxxxx
	#define QOI_DIFF_8  0x80 // 10xxxxxx
	#define QOI_DIFF_16 0xc0 // 110xxxxx
	#define QOI_DIFF_24 0xe0 // 1110xxxx
	#define QOI_COLOR   0xf0 // 1111xxxx
	
	#define QOI_MASK_2  0xc0 // 11000000
	#define QOI_MASK_3  0xe0 // 11100000
	#define QOI_MASK_4  0xf0 // 11110000
	"""

	"""
	# 1 bit has to be delta
	TODO 0xxxxxxx

	# 2 bits has to be jump
	0x00 // 00xxxxxx
	0x40 // 01xxxxxx
	TODO 0x80 // 10xxxxxx

	# One of these can be 12-bit full header
	0xc0 // 1100xxxx
	0xd0 // 1101xxxx
	TODO: 0xe0 // 1110xxxx
	TODO: 0xf0 // 1111xxxx

	"""

	"""
	#define QOI_INDEX   0x00 // 00xxxxxx
	#define QOI_DIFF_8  0x80 // 10xxxxxx
	#define QOI_DIFF_16 0xc0 // 110xxxxx
	#define QOI_DIFF_24 0xe0 // 1110xxxx
	#define QOI_COLOR   0xf0 // 1111xxxx
	"""

	TAG_DELTA = 0x00   # 0-------
	TAG_JUMP = 0x80    # 10------
	TAG_RUN = 0xc0     # 110-----
	TAG_FULL = 0xf0    # 1111----

	MASK_DELTA = 0x80  # 1-------
	MASK_JUMP = 0xc0   # 11------
	MASK_RUN = 0xe0    # 111-----
	MASK_FULL = 0xf0   # 1111----

	"""
	# ARCHIVE
	TAG_MASK = 0xc0       # 11000000
	DATA_MASK = ~TAG_MASK # 00111111

	TAG_DELTA = 0x00     # 0xxxxxxx
	DELTA_MASK = 0x80    # 1xxxxxxx

	# 01xxxxxx (0x40) NOT ALLOWED

	TAG_12_FULL = 0x80     # 10xxxxxx
	TAG_RUN = 0xc0       # 11xxxxxx
	
	# TAG_FULL_OLD = 0xfe  # 11111110
	# TAG_FULL = 0xff      # 11111111
	"""

def unsign(x, n_bits):
	max_value = 2 ** n_bits # same as 1 << n_bits
	return (x + max_value) % max_value

def signed(x, n_bits):
	max_value = 2 ** n_bits
	if x > max_value / 2:
		x -= max_value
	return x

def rescale(value):
	return (value << 4) | (value >> 8)

def unscale(value):
	return value >> 4

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

class ByteWriter:

	def __init__(self):
		self.header = bytearray()
		self.data = bytearray()

	def write_header(self, byte: int):
		self.header.append(byte % 256)

	def write_2_bytes_header(self, value):
		self.write_header((0x0000ff00 & value) >> 8)
		self.write_header((0x000000ff & value))

	def write_4_bytes_header(self, value):
		self.write_header((0xff000000 & value) >> 24)
		self.write_header((0x00ff0000 & value) >> 16)
		self.write_header((0x0000ff00 & value) >> 8)
		self.write_header((0x000000ff & value))

	def write(self, byte: int):
		self.data.append(byte % 256)

	def write_2_bytes(self, value):
		self.write((0x0000ff00 & value) >> 8)
		self.write((0x000000ff & value))
	
	def set_data(self, data):
		self.data = data

	def output_header(self):
		return bytes(self.header)
	
	def output_data(self):
		return bytes(self.data)

	def output(self):
		return bytes(self.header + self.data)


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

		self.width, self.height = image.shape
		self.size = self.width * self.height

		self.writer = ByteWriter()
		self.out_path = out_path

		self.stats = [['Section', 'Size (KB)', 'Ratio (x)']]
		self.info = defaultdict(int)

		self.block_size = 16

	@property
	def MAGIC(self):
		a, b, c, d = self.config['magic']
		return ord(a) << 24 | ord(b) << 16 | ord(c) << 8 | ord(d)

	def write_header(self):

		# Write magic
		self.writer.write_4_bytes_header(self.MAGIC)

		# Image dimensions
		self.writer.write_2_bytes_header(self.width)
		self.writer.write_2_bytes_header(self.height)

		# Color bit depth format
		self.writer.write_header(self.config['channels'])
		self.writer.write_header(self.config['bytes_per_channel'])

		# Encoding configuration
		self.writer.write_header(int(self.config['fractal_transform']))
		self.writer.write_header(int(self.config['deflate_compression']))
		self.writer.write_header(int(self.config['segmentation_transform']))

	def encode(self):

		if self.config['verbose']:
			print(f'[QOI CORE ENCODER FORMAT]')

		self.raw_size = self.size * self.config['channels'] * self.config['bytes_per_channel']
		if self.raw_size > 400_000_000_000:
			raise MemoryError(f"Maximum byte count exceeded: {self.raw_size}")

		if not self.config['delta_transform']:
			raise NotImplementedError("Non-delta encoding not supported")

		self.stats.append(['Original', self.raw_size / 1000, 1.0])
		
		# Writing header
		self.write_header()

		pixel_jump = self.config['channels'] * self.config['bytes_per_channel']

		if self.config['fractal_transform']:
			self.curve = GeneralizedHilbertCurve(self.width, self.height, get_index = True)
			# pixel_order = self.curve.generator()
			pixel_order = self.curve.generate_all()
		else:
			pixel_order = range(self.size)

		# if self.config['zipper_transform']:
		# 	pixel_order = self.curve.zipper_transform(pixel_order)

		if self.config['segmentation_transform']:

			# Getting initial reordered
			pixels = self.image.flatten().tolist()
			data = [pixels[i] for i in pixel_order]

			# Setting up reorganizer
			self.partition = BlockPartitioner(data, block_size = self.block_size)
			self.partition.set_delta_changes_array()
			self.partition.initial_partition()

			# Reorganizing pixels based on block partitioning algorithm
			pixel_order, block_jumps = self.partition.block_partition()
			print(block_jumps)

		blocks_written = set()

		n = -1
		run = 0
		block = 0

		prev_pixel = Pixel()
		curr_pixel = Pixel()

		for i in pixel_order:

			n += 1

			index = pixel_jump * i
			px = self.image_bytes[index : index + pixel_jump]
			block = int(i / self.block_size)

			# print(f'n={n} | i={i} | idx={index} | block={block}')
			
			# QUERY CLUSTER JUMPING
			if self.config['segmentation_transform'] and block in block_jumps and block not in blocks_written:
				jump = block_jumps[block] - block
				print(f'i: {i} | block {block} -> block {block_jumps[block]} (jump = {jump})')
				self.writer.write(Utils.TAG_JUMP | jump)
				blocks_written.add(block)

			prev_pixel.update(curr_pixel.bytes)
			curr_pixel.update(px)

			# Run length encoding
			if curr_pixel == prev_pixel:
				run += 1
				if run == 32 or (n + 1) >= self.size:
					self.writer.write(Utils.TAG_RUN | (run - 1))
					run = 0
				continue

			if run:
				self.info['run'] += 1
				self.writer.write(Utils.TAG_RUN | (run - 1))
				run = 0

			# Delta between current and previous pixel
			delta = curr_pixel.value - prev_pixel.value

			# Near delta encoding
			if -64 < delta < 65:
				self.info['delta'] += 1
				self.writer.write(Utils.TAG_DELTA | unsign(delta, 7))
				continue

			# Full pixel delta encoding
			self.info['full'] += 1
			# self.writer.write(Utils.TAG_FULL)
			self.writer.write_2_bytes((Utils.TAG_FULL << 8) | unsign(delta, 12))

		if self.config['verbose']:
			print('\n' + json.dumps(self.info))

		# Write EOF termination
		if self.config['end_of_file'] is not None:
			self.writer.write(self.config['end_of_file'])

		output = self.writer.output()

		ratio = len(self.image_bytes) / len(output)
		self.stats.append(['Initial', len(output) / 1000, ratio])

		if self.config['deflate_compression']:

			data = self.writer.output_data()
			compressed = zlib.compress(data, level = 9)

			zlib_ratio = len(output) / len(compressed)
			self.stats.append(['DEFLATE', len(compressed) / 1000, zlib_ratio])

			self.writer.set_data(compressed)

		# if self.config['aes_encryption']:

		# 	data = self.writer.output_data()
		# 	encrypted = Encrypt(data, self.config['secret_key'])

		# 	self.writer.set_data(encrypted)

		output = self.writer.output()
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

		self.reader = ByteReader(self.file_bytes)
		self.out_path = out_path

		self.block_size = 16

		self.fulls = []

	@property
	def MAGIC(self):
		a, b, c, d = self.config['magic']
		return ord(a) << 24 | ord(b) << 16 | ord(c) << 8 | ord(d)

	def read_header(self):

		header_magic = self.reader.read_4_bytes()
		if header_magic != self.MAGIC:
			raise ValueError('Image does not contain valid header')

		self.width = self.reader.read_2_bytes()
		self.height = self.reader.read_2_bytes()

		self.channels = self.reader.read()
		self.bytes_per_channel = self.reader.read()
		
		self.fractal_transform = bool(self.reader.read())
		# self.fractal_transform = False
		
		self.deflate_compression = bool(self.reader.read())
		self.segmentation_transform = bool(self.reader.read())

	def decode(self):

		self.read_header()

		pixel_jump = self.channels * self.bytes_per_channel

		self.size = self.width * self.height
		self.total_size = self.width * self.height * pixel_jump

		self.pixel_data = bytearray(self.total_size)
		
		# Continue reading after header
		if self.deflate_compression:
			self.reader = ByteReader(zlib.decompress(self.file_bytes[self.reader.read_pos:]))
		
		# i = 0
		# while i < len(self.reader.bytes):
		# 	data = self.reader.bytes[i]
		# 	block = int(i / self.block_size)
		# 	if (data & Utils.MASK_JUMP) == Utils.TAG_JUMP:
		# 		jump = (~Utils.MASK_JUMP) & data
		# 		print(f'i = {i} | block {block} jump: {jump} to {block + jump}')
		# 		# i += 1
		# 		if (Utils.MASK_FULL & data) == Utils.TAG_FULL:
		# 			i += 1
		# 	i += 1

		if self.fractal_transform:
			self.curve = GeneralizedHilbertCurve(self.width, self.height, get_index = True)
			pixel_order = self.curve.generate_all()
		else:
			pixel_order = range(self.size)

		block_pixel_orders = np.asarray(pixel_order).reshape((self.size // self.block_size, self.block_size))
		
		n = -1
		index = -1
		run = 0

		pixel = Pixel()
		prev_pixel = Pixel()

		for i in pixel_order:

			block = int(i / self.block_size)
			n += 1

			if index >= 0:
				self.pixel_data[index : index + pixel_jump] = pixel.bytes
				prev_pixel.update(pixel.bytes)

			index = pixel_jump * i

			if run > 0:
				# self.fulls.append(i)
				run -= 1
				continue

			data = self.reader.read()
			if data is None:
				break

			# Next 32 pixel indexes are all messed up
			if (data & Utils.MASK_JUMP) == Utils.TAG_JUMP:
				jump = (~Utils.MASK_JUMP) & data
				# print(f'block {block} jump: {jump} to {block + jump}')
				data = self.reader.read()
		
			if (data & Utils.MASK_FULL) == Utils.TAG_FULL:
				
				rest = self.reader.read()
				full_data = data << 8 | rest

				delta = signed(full_data & 0xFFF, 12)

				recovered = (prev_pixel.value + delta)
				pixel.update(recovered.to_bytes(2, sys.byteorder))
				self.fulls.append(i)

				continue

			if (data & Utils.MASK_DELTA) == Utils.TAG_DELTA:
				delta = signed(~Utils.MASK_DELTA & data, 7) # signed(data, 7)
				recovered = (prev_pixel.value + delta)
				pixel.update(recovered.to_bytes(2, sys.byteorder))
				# self.fulls.append(i)
				continue

			if (data & Utils.MASK_RUN) == Utils.TAG_RUN:
				# run = (data & 0x3f)
				run = (data & ~Utils.MASK_RUN)
				# self.fulls.append(i)

		if self.out_path is not None:

			# Scale from 12 bit image to 16 bit display
			pixels = np.frombuffer(bytes(self.pixel_data), dtype = 'uint16').reshape(self.width, self.height)
			preview = np.vectorize(rescale)(pixels).astype('uint16')

			# preview = preview.flatten()
			# preview = np.zeros(self.size, dtype = np.uint16)
			# preview[self.fulls] = 60000
			# preview = preview.reshape(self.width, self.height)

			# preview[:, ::8] = 30000
			# preview[::8, :] = 30000

			# NOTE: Writing to PNG is a time bottleneck
			import imageio
			imageio.imwrite(self.out_path, preview)

			return pixels

		return self.pixel_data
	
