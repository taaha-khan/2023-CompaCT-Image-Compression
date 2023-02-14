
def wrap(x):
	return (x + (1 << 6)) % 64

def back(x):
	if x > 32:
		x -= 64
	return x

def delta_transform(data):
	deltas = []
	deltas.append(data[0])
	for i in range(1, len(data)):
		delta = data[i] - data[i - 1]
		unsigned = (delta + (1 << 8)) % 256
		deltas.append(unsigned)
	return deltas

def revert_delta_transform(data):
	output = []
	output.append(data[0])
	for i in range(1, len(data)):
		delta = data[i]
		if delta > 127:
			delta -= 256
		recovered = (output[i - 1] + delta) % 256
		output.append(recovered)
	return output

def rescale(value):
	return (value << 4) | (value >> 8)

def unscale(value):
	return value >> 4

if __name__ == '__main__':

	from Crypto.Cipher import AES
	from Crypto.Random import get_random_bytes

	key = get_random_bytes(24)
	print(key.hex())

	data = b'data'

	cipher = AES.new(key, AES.MODE_CFB)
	cipher_text = cipher.encrypt(data)

	print(cipher_text.hex())

	decrypt = decrypt_cipher = AES.new(key, AES.MODE_CFB, iv = cipher.iv)
	output = decrypt_cipher.decrypt(cipher_text)

	print(output.hex())

	exit()

	import numpy as np
	arr = np.arange(4096)

	print(arr)

	arrB = np.vectorize(rescale)(arr)

	print(arrB)

	arrC = np.vectorize(unscale)(arrB)

	print(arrC)

	exit()


	# for A in range(4096):

	# A = 1095
	# print(A, bin(A)[2:])

	B = A << 4
	# print(B, bin(B)[2:])

	C = A >> (12 - 4)

	# print(C, bin(C)[2:])

	D = B | C
	# print(A, D, bin(D)[2:])
	# print(A, B, bin(B)[2:])

	E = D >> 4
	# print(E, bin(E)[2:])

	print(arr)

	exit()

	value = 3200
	header = 0x80

	print('header', bin(header)[2:])
	print('value', bin(value)[2:])

	a = (header << 8) | value

	b = bin(a)[2:]
	print('together', b)
	print('len', len(b))

	# mask = 0b111111111111
	mask = 0xFFF

	c = (a << 12) >> 12
	c = a & mask
	print(bin(c)[2:])
	print('cut', c)

	exit()

	import imageio
	import numpy as np

	# Construct 16-bit gradient greyscale image
	im = np.arange(65536, dtype = np.uint16).reshape(256, 256)
	# im = np.arange(256, dtype = np.uint8).reshape(16, 16)
	imageio.imwrite('data/result.png', im) 

	exit()

	import glob

	dataset_directory = 'C:/Users/taaha/Downloads/manifest-OtXaMwL56190865641215613043/QIN LUNG CT/'

	a = set(glob.glob(dataset_directory + '**/*.dcm', recursive = True))
	print(len(a))

	exit()

	for i in range(-128, 128, 1):
		w = wrap(i)
		b = back(wrap(i))
		print(f'{i} -> {w} -> {b} == {i == b}')

	exit()

	import random
	random.seed(0)

	data = [1, 2, 3, 3, 3, 3, 4, 2, 1, 0]
	# data = [random.randrange(256) for i in range(20)]
	print(data)

	a = delta_transform(data)
	print(a)

	b = revert_delta_transform(a)
	print(b)
