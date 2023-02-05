
import numpy as np
import imageio

def array_to_png(array, path):
	assert path.endswith('png')
	# Scale from 12 bit image to 16 bit display
	preview = array * 16
	imageio.imwrite(path, preview)

def png_to_array(path):
	assert path.endswith('png')
	# Scale from 16 bit display to 12 bit image
	png_pixels = imageio.v2.imread(path)
	scaled = (png_pixels / 16).astype('uint16')
	return scaled

if __name__ == '__main__':

	png_in = 'data/decoded-testing.png'
	png_out = 'data/decoded-png.png'

	a = png_to_array(png_in)
	array_to_png(a, png_out)

	import os

	raw_size = len(a.tobytes())
	print(f'raw pixel data size: {raw_size / 1000} KB')

	size = os.path.getsize(png_out)
	print(f'{png_out} size: {size / 1000} KB')

	ratio = raw_size / size
	print(f'PNG compression ratio: {ratio:.2f}x')
