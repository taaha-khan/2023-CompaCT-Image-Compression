
import sys
sys.dont_write_bytecode = True

from PIL import Image
import numpy as np
import hashlib
import pydicom
import argparse
import time
import json
import os

from codec.core import Encoder, Decoder

def get_filename(path, is_encoding, config):
	
	path, filename = os.path.split(path)
	name, filetype = filename.split('.')

	transfer_type = 'encoded' if is_encoding else 'decoded'
	filetype = config['extension'] if is_encoding else config['decode_format']

	renamed = f'{path}/{transfer_type}-{name}.{filetype}'
	return renamed

def main():
	
	parser = argparse.ArgumentParser()
	parser.add_argument('-e', '--encode', action = 'store_true', default = False)
	parser.add_argument('-d', '--decode', action = 'store_true', default = False)
	parser.add_argument('-f', '--file-path', type = str, required = False)
	parser.add_argument('-o', '--out-path', type = str, required = False)
	args = parser.parse_args()

	directory = os.getcwd().split('\\')[-1]
	config = json.load(open(('src/' if directory != 'src' else '') + 'config.json', 'r'))
	# print(json.dumps(config))

	DEMO_IMAGE_PATH = "C:/Users/taaha/Downloads/manifest-OtXaMwL56190865641215613043/QIN LUNG CT/QIN-LSC-0055/07-27-2003-1-CT Thorax wo Contrast-86597/5.000000-THORAX WO  3.0  B41 Soft Tissue-77621/1-016.dcm"
	
	input_path = args.file_path
	if args.file_path == None or args.decode:
		print(f'File not found, running demo')
		input_path = DEMO_IMAGE_PATH
	
	image = pydicom.read_file(input_path).pixel_array

	if args.encode:

		out_path = get_filename(input_path, True, config)

		encoder = Encoder(config, image, out_path)
		encoder.encode_qoi()
		# encoder.encode_packbits()

		print(f'\n\"{input_path}\" encoded to \"{out_path}\"')

	elif args.decode:

		with open(args.file_path, 'rb') as encoded:
			file_bytes = encoded.read()

		out_path = get_filename(args.file_path, False, config)

		decoder = Decoder(config, file_bytes, out_path)
		output = decoder.decode_qoi()

		print(f'\"{args.file_path}\" preview decoded to \"{out_path}\"')

		# Confirming that reconstruction error is 0
		# error_matrix = image - output
		# error = np.count_nonzero(error_matrix)
		# print(f'\nReconstruction Error: {error}')

		# original_hash = hashlib.sha1(image.tobytes()).hexdigest()
		# recovered_hash = hashlib.sha1(output.tobytes()).hexdigest()

		# print(f'SHA1 Original Hash:  {original_hash}')
		# print(f'SHA1 Recovered Hash: {recovered_hash}')


if __name__ == '__main__':
	main()
