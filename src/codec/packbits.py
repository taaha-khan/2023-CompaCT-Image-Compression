# Adapted from https://github.com/psd-tools/packbits 2013

"""
Copyright (c) 2013 Mikhail Korobov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import sys

class PackBits:

	MAX_LENGTH = 127

	def __init__(self, apply_delta_transform = False):

		self.apply_delta_transform = apply_delta_transform

		self.result = bytearray()

		self.buf = bytearray()
		self.state = 'RAW'
		self.run = 0

		self.pos = 0

	def delta_transform(self, data):
		deltas = []
		deltas.append(data[0])
		for i in range(1, len(data)):
			delta = data[i] - data[i - 1]
			unsigned = (delta + (1 << 8)) % 256
			deltas.append(unsigned)
		return deltas
	
	def revert_delta_transform(self, data):
		output = []
		output.append(data[0])
		for i in range(1, len(data)):
			delta = data[i]
			if delta > 127:
				delta -= 256
			recovered = (output[i - 1] + delta) % 256
			output.append(recovered)
		return output

	def finish_raw(self):
		if len(self.buf) == 0:
			return
		self.result.append(len(self.buf) - 1)
		self.result.extend(self.buf)
		self.buf = bytearray()

	def finish_rle(self):
		self.result.append(256 - (self.run - 1))
		self.result.append(self.data[self.pos])

	def encode(self, data):
		"""
		Encodes data using PackBits encoding.
		"""

		if len(data) == 0:
			return data

		if len(data) == 1:
			return b'\x00' + data

		if self.apply_delta_transform:
			data = self.delta_transform(data)

		self.data = bytearray(data)

		while self.pos < len(self.data) - 1:
			
			if self.data[self.pos] == self.data[self.pos + 1]:
				
				if self.state == 'RAW':
					# end of RAW data
					self.finish_raw()
					self.state = 'RLE'
					self.run = 1

				elif self.state == 'RLE':
					if self.run == self.MAX_LENGTH:
						# restart the encoding
						self.finish_rle()
						self.run = 0
					# move to next byte
					self.run += 1

			else:

				if self.state == 'RLE':
					self.run += 1
					self.finish_rle()
					self.state = 'RAW'
					self.run = 0

				elif self.state == 'RAW':
					if len(self.buf) == self.MAX_LENGTH:
						# restart the encoding
						self.finish_raw()
					self.buf.append(self.data[self.pos])

			self.pos += 1

		if self.state == 'RAW':
			self.buf.append(self.data[self.pos])
			self.finish_raw()
		else:
			self.run += 1
			self.finish_rle()

		return self.result
		# return bytes(self.result)
		
	def decode(self, data):
		"""
		Decodes a PackBit encoded data.
		"""

		self.data = bytearray(data)
		self.result = bytearray()

		self.pos = 0

		while self.pos < len(self.data):

			header = self.data[self.pos]
			if header > 127:
				header -= 256
			self.pos += 1

			if 0 <= header <= 127:
				self.result.extend(self.data[self.pos : self.pos + header + 1])
				self.pos += header + 1
			elif header == -128:
				pass
			else:
				self.result.extend([self.data[self.pos]] * (1 - header))
				self.pos += 1

		if self.apply_delta_transform:
			self.result = self.revert_delta_transform(self.result)

		return self.result
		# return bytes(self.result)

if __name__ == '__main__':

	data = [3, 255, 3, 255, 3, 255, 3, 255, 20, 255]
	print(data)

	pack = PackBits()

	encoded = pack.encode(data)
	print(encoded)

	decoded = pack.decode(encoded)
	print(decoded)