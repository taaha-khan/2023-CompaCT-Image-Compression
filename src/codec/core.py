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

from PIL import Image
import json
import zlib

# from codec.curve import GeneralizedHilbertCurve

class Utils:

	CHANNELS = 3

	# chunk tags
	TAG_INDEX = 0x00  # 00xxxxxx
	TAG_DIFF = 0x40  # 01xxxxxx
	TAG_LUMA = 0x80  # 10xxxxxx
	TAG_RUN = 0xc0  # 11xxxxxx
	TAG_RGB = 0xfe  # 11111111 # FIXME
	TAG_MASK = 0xc0  # 11000000

class Pixel:

	def __init__(self):
		self.px_bytes = bytearray((0, 0, 0))

	def update(self, values: bytes) -> None:
		self.px_bytes[0:3] = values

	def __str__(self) -> str:
		r, g, b = self.px_bytes
		return f'[R = {r}, G = {g}, B = {b}]'

	def __eq__(self, other):
		return self.px_bytes == other.px_bytes

	@property
	def bytes(self) -> bytes:
		return bytes(self.px_bytes)

	@property
	def hash(self) -> int:
		r, g, b = self.px_bytes
		return (r * 3 + g * 5 + b * 7) % 64

	@property
	def red(self) -> int:
		return self.px_bytes[0]

	@property
	def green(self) -> int:
		return self.px_bytes[1]

	@property
	def blue(self) -> int:
		return self.px_bytes[2]

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

	def output(self):
		return bytes(self.bytes[0:self.read_pos])

class Encoder:

	def __init__(self, config, image, out_path):

		self.config = config

		self.image = image
		self.image_bytes = image.tobytes()

		self.width, self.height = image.size
		self.total_size = self.width * self.height

		self.writer = ByteWriter()
		self.out_path = out_path

		self.hash_array = [Pixel() for _ in range(64)]

		self.info = {
			'cache': 0,
			'diff': 0,
			'luma': 0,
			'run': 0,
			'full': 0
		}

	@property
	def MAGIC(self):
		a, b, c, d = self.config['magic']
		return ord(a) << 24 | ord(b) << 16 | ord(c) << 8 | ord(d)

	def encode(self):

		max_n_bytes = 14 + self.total_size * 4 + len(self.config['EOF'])
		if max_n_bytes > 400_000_000:
			raise MemoryError(f"Maximum byte count exceeded: {max_n_bytes}")
		
		# write header
		self.writer.write_4_bytes(self.MAGIC)
		self.writer.write_4_bytes(self.width)
		self.writer.write_4_bytes(self.height)
		self.writer.write(self.config['channels']) # N_CHANNELS = 3
		self.writer.write(0) # srgb colorspace

		# encode pixels
		run = 0

		prev_pixel = Pixel()
		curr_pixel = Pixel()

		pixel_data = (
			self.image_bytes[i:i + self.config['channels']] for i in range(0, len(self.image_bytes), self.config['channels'])
		)

		for i, px in enumerate(pixel_data):

			prev_pixel.update(curr_pixel.bytes)
			curr_pixel.update(px)

			if curr_pixel == prev_pixel:
				run += 1
				if run == 62 or (i + 1) >= self.total_size:
					self.info['run'] += 1
					self.writer.write(Utils.TAG_RUN | (run - 1))
					run = 0
				continue

			if run:
				self.info['run'] += 1
				self.writer.write(Utils.TAG_RUN | (run - 1))
				run = 0

			index_pos = curr_pixel.hash
			if self.hash_array[index_pos] == curr_pixel:
				self.info['cache'] += 1
				self.writer.write(Utils.TAG_INDEX | index_pos)
				continue

			self.hash_array[index_pos].update(curr_pixel.bytes)

			vr = curr_pixel.red - prev_pixel.red
			vg = curr_pixel.green - prev_pixel.green
			vb = curr_pixel.blue - prev_pixel.blue

			vg_r = vr - vg
			vg_b = vb - vg

			if all(-3 < x < 2 for x in (vr, vg, vb)):
				self.info['diff'] += 1
				self.writer.write(Utils.TAG_DIFF | (vr + 2) << 4 | (vg + 2) << 2 | (vb + 2))
				continue

			elif all(-9 < x < 8 for x in (vg_r, vg_b)) and -33 < vg < 32:
				self.info['luma'] += 1
				self.writer.write(Utils.TAG_LUMA | (vg + 32))
				self.writer.write((vg_r + 8) << 4 | (vg_b + 8))
				continue

			self.info['full'] += 1
			self.writer.write(Utils.TAG_RGB)
			self.writer.write(curr_pixel.red)
			self.writer.write(curr_pixel.green)
			self.writer.write(curr_pixel.blue)

		print(json.dumps(self.info))

		# Write EOF termination
		for end in self.config['EOF']:
			self.writer.write(end)

		output = self.writer.output()

		if self.config['zlib_compression']:
			print(f'Before zlib: {len(output)} bytes')
			output = zlib.compress(output, 
				level = self.config['zlib_compression_level'])
			print(f' After zlib: {len(output)} bytes')

		with open(self.out_path, 'wb') as fout:
			fout.write(output)

class Decoder:

	def __init__(self, config, file_bytes, out_path):

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

	def decode(self):

		header_magic = self.reader.read_4_bytes()

		if header_magic != self.MAGIC:
			raise ValueError('provided image does not contain proper header')

		self.width = self.reader.read_4_bytes()
		self.height = self.reader.read_4_bytes()
		self.channels = self.reader.read()
		self.colorspace = self.reader.read()

		self.size = (self.width, self.height)
		self.out_size = self.width * self.height * self.channels

		self.pixel_data = bytearray(self.out_size)

		run = 0
		pixel = Pixel()

		for i in range(-self.channels, self.out_size, self.channels):

			index_pos = pixel.hash
			self.hash_array[index_pos].update(pixel.bytes)

			if i >= 0:
				self.pixel_data[i:i + self.channels] = pixel.bytes

			if run > 0:
				run -= 1
				continue

			data = self.reader.read()
			if data is None:
				break

			if data == Utils.TAG_RGB:
				new_value = bytes((self.reader.read() for _ in range(3)))
				pixel.update(new_value)
				continue

			if (data & Utils.TAG_MASK) == Utils.TAG_INDEX:
				pixel.update(self.hash_array[data].bytes)
				continue

			if (data & Utils.TAG_MASK) == Utils.TAG_DIFF:
				red = (pixel.red + ((data >> 4) & 0x03) - 2) % 256
				green = (pixel.green + ((data >> 2) & 0x03) - 2) % 256
				blue = (pixel.blue + (data & 0x03) - 2) % 256
				pixel.update(bytes((red, green, blue)))
				continue

			if (data & Utils.TAG_MASK) == Utils.TAG_LUMA:
				b2 = self.reader.read()
				vg = ((data & 0x3f) % 256) - 32
				red = (pixel.red + vg - 8 + ((b2 >> 4) & 0x0f)) % 256
				green = (pixel.green + vg) % 256
				blue = (pixel.blue + vg - 8 + (b2 & 0x0f)) % 256
				pixel.update(bytes((red, green, blue)))
				continue

			if (data & Utils.TAG_MASK) == Utils.TAG_RUN:
				run = (data & 0x3f)

		image = Image.frombuffer('RGB', self.size, bytes(self.pixel_data), 'raw')
		image.save(self.out_path, 'png')
