
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

if __name__ == '__main__':

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
