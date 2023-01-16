
import sys
import os

sys.dont_write_bytecode = True
sys.path.append(os.path.join('scripts', '..', 'src'))

import json
import time
import random

import concurrent.futures as processor
from tabulate import tabulate
from tqdm import tqdm

import pydicom
from PIL import Image

from codec.core import Encoder, Decoder

def new_compressor(path, config):

	FILE = 'File'
	RATIO = 'Ratio (x)'
	SAVED = 'Space Saved (%)'
	TIME = 'Time (s)'

	r = random.uniform(0.0, 10.0)

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

	# ds.compress(pydicom.uid.RLELossless, image) #, encoding_plugin = 'pylibjpeg')
	# tmp_saved = f'C:/Users/taaha/Downloads/rle_ct_dataset/{os.path.basename(path)}'
	# ds.save_as(tmp_saved)
	# compressed_size = os.path.getsize(tmp_saved) # bytes

	encoder = Encoder(config, image, None)
	compressed = encoder.encode_packbits()
	compressed_size = len(compressed) # bytes

	output[TIME] = time.process_time() - start
	output[RATIO] = original_size / compressed_size
	output[SAVED] = 100 - (100 / output[RATIO])

	return output

def main():

	directory = os.getcwd().split('\\')[-1]
	config = json.load(open('src/config.json', 'r'))
	config['verbose'] = False

	dataset_directory = 'C:/Users/taaha/Downloads/ct_nonequi_tilt/'

	processes = []
	outputs = []
	
	with processor.ProcessPoolExecutor() as executor:

		for filename in os.listdir(dataset_directory):
			if not filename.endswith('dcm'):
				continue
			path = dataset_directory + filename
			processes.append(executor.submit(new_compressor, path, config))

		print(f'Queued to compress {len(processes)} testing images on {os.cpu_count()} threads')

		with tqdm(total = len(processes)) as bar:
			for process in processor.as_completed(processes):
				outputs.append(process.result())
				bar.update(1)

	print('\n')

	table = tabulate(outputs, headers = 'keys', tablefmt = 'simple_outline') # .replace('-', '━')
	print(table)


if __name__ == '__main__':
	main()