
import sys
import os

sys.dont_write_bytecode = True
sys.path.append(os.path.join('scripts', '..', 'src'))
sys.path.append(os.path.join('scripts', '..', 'lib'))

from collections import defaultdict
import json
import time
import random
import glob

import concurrent.futures as processor
from tabulate import tabulate
from tqdm import tqdm

import pydicom

# Comparison standards
from png import array_to_png
from jpeg2000 import png_to_jpeg2000

# Proposed encoder
from codec.core import Encoder

# Data keys
FILE = 'File'
RAW = 'Raw'
PNG = 'PNG'
JP2 = 'JP2'
RLE = 'RLE'
KHN = 'KHN'

# Directories 
# dataset_directory = 'C:/Users/taaha/Downloads/ct_nonequi_tilt/'
dataset_directory = 'C:/Users/taaha/Downloads/manifest-OtXaMwL56190865641215613043/QIN LUNG CT/R0223/12-05-2001-NA-CT CHEST WITH CONTRAST-15336/2.000000-NA-08982/'
# dataset_directory = 'C:/Users/taaha/Downloads/manifest-OtXaMwL56190865641215613043/QIN LUNG CT/'
results_file = 'results/encoder-comparisons.csv'
temp_directory = 'C:/Users/taaha/Downloads/temp-encoded-dataset'

def get_temp_file_path(filepath, uid, extension):
	input_path, input_file = os.path.split(filepath)
	_, input_extension = os.path.splitext(input_file)
	output_file = input_file.replace(input_extension, extension)
	output_path = f'{temp_directory}/({uid:04})-{output_file}'
	return output_path

def comparison(path, config, uid = None):

	ufile = f'({uid:04})-{os.path.basename(path)}'

	# Defaults for debugging
	output = dict.fromkeys([FILE, RAW, PNG, JP2, RLE, KHN], 1)
	output[FILE] = ufile
	# return output

	ds = pydicom.read_file(path)
	image = ds.pixel_array

	# RAW
	raw_size = len(ds.PixelData)
	output[RAW] = raw_size
	
	# PNG
	png_saved = get_temp_file_path(path, uid, '.png')
	array_to_png(image, png_saved)
	output[PNG] = os.path.getsize(png_saved)

	# JPEG2000 LOSSLESS
	jp2_saved = get_temp_file_path(path, uid, '.jp2')
	png_to_jpeg2000(png_saved, jp2_saved)
	output[JP2] = os.path.getsize(jp2_saved)

	# BUILTIN RLE
	ds.compress(pydicom.uid.RLELossless, image) # , encoding_plugin = 'pylibjpeg')
	output[RLE] = len(ds.PixelData)

	# PROPOSED
	encoder = Encoder(config, image, out_path = None)
	compressed = encoder.encode_qoi()
	output[KHN] = len(compressed)

	return output

def main():

	config = json.load(open('src/config.json', 'r'))
	config['verbose'] = False

	processes = []
	outputs = []
	
	with processor.ProcessPoolExecutor() as executor:

		for n, filename in enumerate(glob.glob(dataset_directory + '**/*.dcm', recursive = True)):
			# if os.path.basename(filename).startswith('1-1'):
			# 	continue

			processes.append(executor.submit(comparison, filename, config, n))
			
			# if n > 200:
			# 	break

		print(f'{len(processes)} testing images queued on {os.cpu_count()} threads')

		with tqdm(total = len(processes)) as bar:
			for process in processor.as_completed(processes):
				outputs.append(process.result())
				bar.update(1)

	outputs.sort(key = lambda a: a[FILE])

	# Dump all comparison results to big boy csv
	with open(results_file, 'w') as fout:
		fout.write(','.join(outputs[0].keys()))
		for line in outputs:
			fout.write('\n' + ','.join(map(str, line.values())))

	table = tabulate(outputs, headers = 'keys', tablefmt = 'simple_outline') # .replace('-', '━')
	print(table)

if __name__ == '__main__':
	main()