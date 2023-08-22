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
import numpy as np
import tqdm
import json
import zlib
import sys

from codec.curve import GeneralizedHilbertCurve
from codec.cluster import BlockPartitioner

# Header byte tags
class Utils:
	
	TAG_DELTA = 0x00   # 0-------
	TAG_JUMP = 0x80    # 10------
	TAG_RUN = 0xc0     # 110-----
	TAG_FULL = 0xe0    # 1110----

	MASK_DELTA = 0x80  # 1-------
	MASK_JUMP = 0xc0   # 11------
	MASK_RUN = 0xe0    # 111-----
	MASK_FULL = 0xf0   # 1111----

def unsign(x, n_bits):
	max_value = 2 ** n_bits # same as 1 << n_bits
	return (x + max_value) % max_value

def signed(x, n_bits):
	max_value = 2 ** n_bits
	if x > max_value / 2:
		x -= max_value
	return x

def rescale(value):
	# return (value << 4) | (value >> 8)
	return (value << 4) | (0 >> 8)

def unscale(value):
	return value >> 4

class Pixel:

	# TODO: Only deal with full value, not bytes

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

	# FIXME: Length of config['encoder']['end_of_file'] 
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

	def peek(self):
		return self.bytes[self.read_pos]

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

	@property
	def MAGIC(self):
		magic = bytes(map(ord, reversed(self.config['magic'])))
		return int.from_bytes(magic, sys.byteorder)

	def write_header(self):

		# Write magic
		self.writer.write_4_bytes_header(self.MAGIC)

		# Image dimensions
		self.writer.write_2_bytes_header(self.width)
		self.writer.write_2_bytes_header(self.height)

		# Color bit depth format
		self.writer.write_header(self.config['encoder']['channels'])
		self.writer.write_header(self.config['encoder']['bytes_per_channel'])

		# Encoding configuration
		self.writer.write_header(int(self.config['encoder']['transforms']['fractal']))
		self.writer.write_header(int(self.config['encoder']['transforms']['segmentation']))
		self.writer.write_header(int(self.config['encoder']['deflate_compression']))
		# self.writer.write_header(int(self.config['encoder']['aes_encryption']))

	def encode(self):

		# if self.config['verbose']:
		# 	print(f'[QOI CORE ENCODER FORMAT]')

		self.raw_size = self.size * self.config['encoder']['channels'] * self.config['encoder']['bytes_per_channel']
		if self.raw_size > 400_000_000_000:
			raise MemoryError(f"Maximum byte count exceeded: {self.raw_size}")

		if not self.config['encoder']['transforms']['delta']:
			raise NotImplementedError("Non-delta encoding not supported")

		if self.config['encoder']['transforms']['zipper']:
			raise NotImplementedError("Zipper transform not supported or encouraged")

		self.stats.append(['Original', self.raw_size / 1000, 1.0])
		
		# Writing header
		self.write_header()

		pixel_jump = self.config['encoder']['channels'] * self.config['encoder']['bytes_per_channel']

		if self.config['encoder']['transforms']['fractal']:
			self.curve = GeneralizedHilbertCurve(self.width, self.height, get_index = True)
			# pixel_order = self.curve.generator()
			pixel_order = self.curve.generate_all()
		else:
			pixel_order = range(self.size)

		if self.config['encoder']['transforms']['zipper']:
			pixel_order = self.curve.zipper_transform(pixel_order)

		blocks_written = set()
		block_pixel_orders = np.asarray(pixel_order).reshape((self.size // self.config['block_size'], self.config['block_size']))
		pixel_block = {}
		for i, block in enumerate(block_pixel_orders):
			for px in block:
				pixel_block[px] = i

		if self.config['encoder']['transforms']['segmentation']:

			# Getting initial reordered
			pixels = self.image.flatten().tolist()
			data = [pixels[i] for i in pixel_order]

			# Block segmentation algorithm
			self.partition = BlockPartitioner(
				data = data, order = pixel_order, 
				block_size = self.config['block_size']
			)

			# Initialization
			self.partition.set_delta_changes_array()
			self.partition.initial_partition()

			# Reorganizing pixels based on block partitioning algorithm
			pixel_order, block_jumps = self.partition.block_partition()
			# print(block_jumps)

			# if self.config['verbose']:
			# 	print(f'{len(block_jumps)} block jumps')

		n = -1
		run = 0
		block = 0

		prev_pixel = Pixel()
		curr_pixel = Pixel()

		for i in pixel_order:

			n += 1

			index = pixel_jump * i
			px = self.image_bytes[index : index + pixel_jump]
			block = pixel_block[i]

			# Query cluster jumping
			if self.config['encoder']['transforms']['segmentation']:
				if block in block_jumps and block not in blocks_written:
					jump = block_jumps[block] - block
					self.writer.write(Utils.TAG_JUMP | jump)
					blocks_written.add(block)

			prev_pixel.update(curr_pixel.bytes)
			curr_pixel.update(px)

			# Run length encoding
			# if curr_pixel == prev_pixel:
			# 	run += 1
			# 	self.info['run'] += 1
			# 	if run == 32 or (n + 1) >= self.size:
			# 		self.writer.write(Utils.TAG_RUN | (run - 1))
			# 		run = 0
			# 	continue

			# if run:
			# 	self.writer.write(Utils.TAG_RUN | (run - 1))
			# 	run = 0

			# Delta between current and previous pixel
			delta = curr_pixel.value - prev_pixel.value

			# Short delta encoding
			if -64 < delta < 65:
				self.info['delta'] += 1
				self.writer.write(Utils.TAG_DELTA | unsign(delta, 7))
				continue

			# Full delta encoding
			self.info['full'] += 1
			self.writer.write_2_bytes((Utils.TAG_FULL << 8) | unsign(delta, 12))

		if self.config['verbose']:
			print('\n' + json.dumps(self.info))

		# Write EOF termination
		if self.config['encoder']['end_of_file'] is not None:
			self.writer.write(self.config['encoder']['end_of_file'])

		output = self.writer.output()

		ratio = self.raw_size / len(output)
		self.stats.append(['QOI', len(output) / 1000, ratio])

		if self.config['encoder']['deflate_compression']:

			data = self.writer.output_data()
			compressed = zlib.compress(data, level = 9)

			zlib_ratio = (len(output) - len(self.writer.header)) / len(compressed)
			self.stats.append(['DEFLATE', (len(self.writer.header) + len(compressed)) / 1000, zlib_ratio])

			self.writer.set_data(compressed)

		# if self.config['encoder']['aes_encryption']:
		# 	data = self.writer.output_data()
		# 	encrypted = Encrypt(data, self.config['secret_key'])
		# 	self.writer.set_data(encrypted)

		output = self.writer.output()
		ratio = self.raw_size / len(output)

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

		self.fulls = []

	@property
	def MAGIC(self):
		magic = bytes(map(ord, reversed(self.config['magic'])))
		return int.from_bytes(magic, sys.byteorder)

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
		self.segmentation_transform = bool(self.reader.read())
		
		self.deflate_compression = bool(self.reader.read())
		# self.aes_encryption = bool(self.reader.read())

	def decode(self):

		self.read_header()

		pixel_jump = self.channels * self.bytes_per_channel

		self.size = self.width * self.height
		self.total_size = self.width * self.height * pixel_jump

		self.pixel_data = bytearray(self.total_size)
		
		# TODO: Decrypt
		# if self.aes_encryption:
		#	self.reader = ByteReader(Decrypt(self.file_bytes[self.reader.read_pos:]))
		
		# Continue reading after header
		if self.deflate_compression:
			self.reader = ByteReader(zlib.decompress(self.file_bytes[self.reader.read_pos:]))

		if self.fractal_transform:
			self.curve = GeneralizedHilbertCurve(self.width, self.height, get_index = True)
			pixel_order = self.curve.generate_all()
		else:
			pixel_order = range(self.size)

		block_pixel_orders = np.asarray(pixel_order).reshape((self.size // self.config['block_size'], self.config['block_size']))

		pixel_block = {}
		for i, block in enumerate(block_pixel_orders):
			for px in block:
				pixel_block[px] = i

		output = np.zeros(self.size, dtype = np.uint16)

		# Populate every other spot with initial pixel order
		padded_order = np.full(2 * self.size, -1)
		padded_order[0::2] = np.asarray(pixel_order)

		completed = np.full(self.size, False)

		running = -1
		n = -1
		index = -1

		run = 0

		pixel = Pixel()
		prev_pixel = Pixel()

		while True:

			running += 1
			if running >= padded_order.size:
				break

			if padded_order[running] == -1:
				continue

			index = padded_order[running]
			if completed[index]:
				continue
			
			n += 1

			block = pixel_block[index]
			completed[index] = True

			# if 598 <= block <= 607:
			# 	print(f'\tn={n} running={running} idx={index} run={run} | {block}')
			
			# if 15620 <= block <= 15622:
			# 	print(f'\tn={n} running={running} idx={index} run={run} | {block}')
			
			# if run > 0:
			# 	run -= 1
			# 	output[index] = pixel.value
			# 	prev_pixel.update(pixel.bytes)
			# 	continue

			# Next 32 pixel indexes are meshed with future
			if (self.reader.peek() & Utils.MASK_JUMP) == Utils.TAG_JUMP:
				
				# Read next encoded data
				data = self.reader.read()
				jump = (~Utils.MASK_JUMP) & data

				# print(f'n={n} running={running} idx={index} run={run} | mesh {block} + {jump:02} = {block + jump}')

				# Populate padded spaces with meshed block
				blockB = block_pixel_orders[block + jump]
				padded_order[running + 1 : running + 1 + 2 * self.config['block_size'] : 2] = blockB

			data = self.reader.read()
		
			# --------------------------------------------------------

			if (data & Utils.MASK_FULL) == Utils.TAG_FULL:
				
				full_data = data << 8 | self.reader.read()
				delta = signed(full_data & 0xFFF, 12)

				recovered = prev_pixel.value + delta
				pixel.update(recovered.to_bytes(2, sys.byteorder))

				self.fulls.append(index)

			# elif (data & Utils.MASK_RUN) == Utils.TAG_RUN:
			# 	run = ~Utils.MASK_RUN & data

			elif (data & Utils.MASK_DELTA) == Utils.TAG_DELTA:
				delta = signed(~Utils.MASK_DELTA & data, 7) # signed(data, 7)
				recovered = prev_pixel.value + delta
				pixel.update(recovered.to_bytes(2, sys.byteorder))

			# Writing pixel to position
			output[index] = pixel.value
			prev_pixel.update(pixel.bytes)

		if self.out_path is not None:

			# Scale from 12 bit image to 16 bit display
			pixels = output.reshape(self.width, self.height)
			preview = np.vectorize(rescale)(pixels).astype('uint16')

			# preview = preview.flatten()
			# preview = np.zeros(self.size, dtype = np.uint16)
			# preview[self.fulls] = 60000
			# preview = preview.reshape(self.width, self.height)

			# preview[:, ::8] = 40000
			# preview[::8, :] = 40000

			# NOTE: Writing to PNG is a time bottleneck
			import imageio
			imageio.imwrite(self.out_path, preview)

			return pixels

		# Return bytes of output
		return output.tobytes()
		# return self.pixel_data
	
