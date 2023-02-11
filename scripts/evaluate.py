
import sys
import os

sys.dont_write_bytecode = True
sys.path.append(os.path.join('scripts', '..', 'src'))

import json
import time
import random
import glob

import concurrent.futures as processor
from tabulate import tabulate
from tqdm import tqdm

import pydicom
from PIL import Image

from png import array_to_png
from codec.core import Encoder, Decoder

FILE = 'File'
RATIO = 'Ratio (x)'
SAVED = 'Space Saved (%)'
TIME = 'Time (s)'

def new_compressor(path, config):

	# r = random.uniform(0.0, 10.0)
	r = 1.0

	output = {
		FILE: os.path.basename(path), 
		RATIO: r,
		SAVED: 100 - (100 / r),
		TIME: random.random()
	}

	# return output

	original_size = os.path.getsize(path) # bytes

	ds = pydicom.read_file(path)
	image = ds.pixel_array

	start = time.process_time()

	# # PNG ENCODER
	tmp_saved = f'C:/Users/taaha/Downloads/rle_ct_dataset/{os.path.basename(path)[:-4]}.png'
	array_to_png(image, tmp_saved)
	compressed_size = os.path.getsize(tmp_saved)

	# # BUILTIN RLE LOSSLESS ENCODER
	# tmp_saved = f'C:/Users/taaha/Downloads/rle_ct_dataset/{os.path.basename(path)}'
	# ds.compress(pydicom.uid.RLELossless, image, encoding_plugin = 'pylibjpeg')
	# ds.save_as(tmp_saved)
	# compressed_size = os.path.getsize(tmp_saved) # bytes

	# # PROPOSED ENCODER
	# encoder = Encoder(config, image, None)
	# compressed = encoder.encode_qoi()
	# # compressed = encoder.encode_packbits()
	# compressed_size = len(compressed) # bytes

	output[TIME] = time.process_time() - start
	output[RATIO] = original_size / compressed_size
	output[SAVED] = 100 - (100 / output[RATIO])

	return output

def main():

	directory = os.getcwd().split('\\')[-1]
	config = json.load(open('src/config.json', 'r'))
	config['verbose'] = False

	# dataset_directory = 'C:/Users/taaha/Downloads/ct_nonequi_tilt/'
	dataset_directory = 'C:/Users/taaha/Downloads/manifest-OtXaMwL56190865641215613043/QIN LUNG CT/R0223/12-05-2001-NA-CT CHEST WITH CONTRAST-15336/2.000000-NA-08982/'
	# dataset_directory = 'C:/Users/taaha/Downloads/manifest-OtXaMwL56190865641215613043/QIN LUNG CT/'

	processes = []
	outputs = []
	
	with processor.ProcessPoolExecutor() as executor:

		for n, filename in enumerate(glob.glob(dataset_directory + '**/*.dcm', recursive = True)):
			# if os.path.basename(filename).startswith('1-1'):
			# 	continue

			processes.append(executor.submit(new_compressor, filename, config))
			
			# if n > 200:
			# 	break

		print(f'{len(processes)} testing images queued on {os.cpu_count()} threads')

		with tqdm(total = len(processes)) as bar:
			for process in processor.as_completed(processes):
				outputs.append(process.result())
				bar.update(1)

	# TODO: Dump all compressor results to big boy csv
	# TODO: Analyze data in jupyter notebook to results folder

	table = tabulate(outputs, headers = 'keys', tablefmt = 'simple_outline') # .replace('-', '━')
	print(table)

	ratios = 0.0
	times = 0.0

	for output in outputs:
		times += output[TIME]
		ratios += output[RATIO]

	times /= len(outputs)
	ratios /= len(outputs)

	info = []
	info.append(['Metric', 'Value'])
	info.append(['Mean ' + RATIO, ratios])
	info.append(['Mean ' + TIME, times])

	table = tabulate(info, headers = 'firstrow', tablefmt = 'simple_outline')
	print(table)

if __name__ == '__main__':
	main()