
import numpy as np
import imageio

def rescale(value):
	# Scale to 16 bits and fill with 4 most significant bits
	return (value << 4) | (value >> 8)

def unscale(value):
	return value >> 4

def array_to_png(array, path):
	assert path.endswith('png')
	# Scale from 12 bit data to 16 bit display
	preview = np.vectorize(rescale)(array).astype('uint16')
	imageio.imwrite(path, preview)

def png_to_array(path):
	assert path.endswith('png')
	# Scale from 16 bit display to 12 bit data
	pixels = imageio.v2.imread(path)
	scaled = np.vectorize(unscale)(pixels).astype('uint16')
	return scaled

if __name__ == '__main__':

	png_in = 'data/working/decoded-testing.png'
	png_out = 'data/working/decoded-png.png'

	a = png_to_array(png_in)
	array_to_png(a, png_out)

	import os

	raw_size = len(a.tobytes())
	print(f'raw pixel data size: {raw_size / 1000} KB')

	size = os.path.getsize(png_out)
	print(f'{png_out} size: {size / 1000} KB')

	ratio = raw_size / size
	print(f'PNG compression ratio: {ratio:.2f}x')
