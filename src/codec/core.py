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

# from codec.curve import GeneralizedHilbertCurve

import argparse
from dataclasses import dataclass, field
from typing import Dict, Optional

from PIL import Image

# chunk tags
TAG_INDEX = 0x00  # 00xxxxxx
TAG_DIFF = 0x40  # 01xxxxxx
TAG_LUMA = 0x80  # 10xxxxxx
TAG_RUN = 0xc0  # 11xxxxxx
TAG_RGB = 0xfe  # 11111111
TAG_MASK = 0xc0  # 11000000


MAGIC = ord('q') << 24 | ord('o') << 16 | ord('i') << 8 | ord('f')

@dataclass
class Pixel:
	px_bytes: bytearray = field(init=False)

	def __post_init__(self):
		self.px_bytes = bytearray((0, 0, 0))

	def update(self, values: bytes) -> None:
		n_channels = len(values)
		if n_channels != 3:
			raise ValueError('a tuple of 3 RGB values should be provided')

		self.px_bytes[0:n_channels] = values

	def __str__(self) -> str:
		r, g, b = self.px_bytes
		return f'R: {r} G: {g} B: {b}'

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

	def __init__(self, size: int):
		self.bytes = bytearray(size)
		self.write_pos = 0

	def write(self, byte: int):
		self.bytes[self.write_pos] = (byte % 256)
		self.write_pos += 1

	def output(self):
		return bytes(self.bytes[0 : self.write_pos])


class ByteReader:
	padding_len = 8

	def __init__(self, data: bytes):
		self.bytes = data
		self.read_pos = 0
		# might be off by 1
		self.max_pos = len(self.bytes) - self.padding_len

	def read(self) -> Optional[int]:
		if self.read_pos >= self.max_pos:
			return None

		out = self.bytes[self.read_pos]
		self.read_pos += 1
		return out

	def output(self):
		return bytes(self.bytes[0:self.read_pos])


def write_32_bits(value: int, writer: ByteWriter) -> None:
	writer.write((0xff000000 & value) >> 24)
	writer.write((0x00ff0000 & value) >> 16)
	writer.write((0x0000ff00 & value) >> 8)
	writer.write((0x000000ff & value))


def read_32_bits(reader: ByteReader) -> int:
	data = [reader.read() for _ in range(4)]
	b1, b2, b3, b4 = data
	return b1 << 24 | b2 << 16 | b3 << 8 | b4


def write_end(writer: ByteWriter) -> None:
	for _ in range(7):
		writer.write(0)
	writer.write(1)


def encode_img(img: Image.Image, srgb: bool, out_path: str) -> None:
	width, height = img.size
	
	img_bytes = img.tobytes()
	output = encode(img_bytes, width, height, srgb)

	with open(out_path, 'wb') as fout:
		fout.write(output)


def decode_to_img(img_bytes: bytes, out_path: str) -> None:
	out = decode(img_bytes)

	size = (out['width'], out['height'])
	img = Image.frombuffer(out['channels'], size, bytes(out['bytes']), 'raw')
	img.save(out_path, 'png')


def encode(img_bytes: bytes, width: int, height: int, srgb: bool):
	total_size = height * width
	channels = 3
	pixel_data = (
		img_bytes[i:i + channels] for i in range(0, len(img_bytes), channels)
	)
	max_n_bytes = 14 + total_size * 4 + 8
	writer = ByteWriter(max_n_bytes)
	hash_array = [Pixel() for _ in range(64)]

	# write header
	write_32_bits(MAGIC, writer)
	write_32_bits(width, writer)
	write_32_bits(height, writer)
	writer.write(3) # N_CHANNELS
	writer.write(0 if srgb else 1)

	# encode pixels
	run = 0
	prev_px_value = Pixel()
	px_value = Pixel()
	for i, px in enumerate(pixel_data):
		prev_px_value.update(px_value.bytes)
		px_value.update(px)

		if px_value == prev_px_value:
			run += 1
			if run == 62 or (i + 1) >= total_size:
				writer.write(TAG_RUN | (run - 1))
				run = 0
			continue

		if run:
			writer.write(TAG_RUN | (run - 1))
			run = 0

		index_pos = px_value.hash
		if hash_array[index_pos] == px_value:
			writer.write(TAG_INDEX | index_pos)
			continue

		hash_array[index_pos].update(px_value.bytes)

		vr = px_value.red - prev_px_value.red
		vg = px_value.green - prev_px_value.green
		vb = px_value.blue - prev_px_value.blue

		vg_r = vr - vg
		vg_b = vb - vg

		if all(-3 < x < 2 for x in (vr, vg, vb)):
			writer.write(TAG_DIFF | (vr + 2) << 4 | (vg + 2) << 2 | (vb + 2))
			continue

		elif all(-9 < x < 8 for x in (vg_r, vg_b)) and -33 < vg < 32:
			writer.write(TAG_LUMA | (vg + 32))
			writer.write((vg_r + 8) << 4 | (vg_b + 8))
			continue

		writer.write(TAG_RGB)
		writer.write(px_value.red)
		writer.write(px_value.green)
		writer.write(px_value.blue)

	write_end(writer)
	return writer.output()


def decode(file_bytes: bytes) -> Dict:
	reader = ByteReader(file_bytes)
	header_magic = read_32_bits(reader)
	
	if header_magic != MAGIC:
		raise ValueError('provided image does not contain proper header')

	width = read_32_bits(reader)
	height = read_32_bits(reader)
	channels = reader.read()
	colorspace = reader.read()

	hash_array = [Pixel() for _ in range(64)]
	out_size = width * height * channels
	pixel_data = bytearray(out_size)
	px_value = Pixel()
	run = 0
	for i in range(-channels, out_size, channels):
		index_pos = px_value.hash
		hash_array[index_pos].update(px_value.bytes)
		if i >= 0:
			pixel_data[i:i + channels] = px_value.bytes

		if run > 0:
			run -= 1
			continue

		b1 = reader.read()
		if b1 is None:
			break

		if b1 == TAG_RGB:
			new_value = bytes((reader.read() for _ in range(3)))
			px_value.update(new_value)
			continue

		if (b1 & TAG_MASK) == TAG_INDEX:
			px_value.update(hash_array[b1].bytes)
			continue

		if (b1 & TAG_MASK) == TAG_DIFF:
			red = (px_value.red + ((b1 >> 4) & 0x03) - 2) % 256
			green = (px_value.green + ((b1 >> 2) & 0x03) - 2) % 256
			blue = (px_value.blue + (b1 & 0x03) - 2) % 256
			px_value.update(bytes((red, green, blue)))
			continue

		if (b1 & TAG_MASK) == TAG_LUMA:
			b2 = reader.read()
			vg = ((b1 & 0x3f) % 256) - 32
			red = (px_value.red + vg - 8 + ((b2 >> 4) & 0x0f)) % 256
			green = (px_value.green + vg) % 256
			blue = (px_value.blue + vg - 8 + (b2 & 0x0f)) % 256
			px_value.update(bytes((red, green, blue)))
			continue

		if (b1 & TAG_MASK) == TAG_RUN:
			run = (b1 & 0x3f)

	out = {
		'width': width, 'height': height,
		'channels': 'RGB',
		'colorspace': colorspace
	}

	out['bytes'] = pixel_data

	return out


def replace_extension(path: str, extension: str) -> str:
	old_extension = path.split('.')[-1]
	new_path = path.replace(old_extension, extension)
	return new_path


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-e', '--encode', action='store_true', default=False)
	parser.add_argument('-d', '--decode', action='store_true', default=False)
	parser.add_argument(
		'-f', '--file-path', type=str,
		help='path to image file to be encoded or decoded', required=True)
	args = parser.parse_args()

	if args.encode:
		try:
			img = Image.open(args.file_path)
		except Exception as exc:
			print(f'image load failed: {exc}')
			return

		out_path = replace_extension(args.file_path, 'qoi')
		encode_img(img, out_path, out_path)

	if args.decode:
		with open(args.file_path, 'rb') as encoded:
			file_bytes = encoded.read()

		out_path = replace_extension(args.file_path, 'png')
		decode_to_img(file_bytes, out_path)


if __name__ == '__main__':
	main()


"""
class Encoder:

	def __init__(self, image_filepath, config):

		self.filepath = image_filepath
		self.config = config

		self.image = None



	def encode(self):
		return

class Decoder:

	def __init__(self, config):
		pass

	def decode(self):
		return
"""