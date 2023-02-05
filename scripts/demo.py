
from PIL import Image
import numpy as np
import hashlib
import pydicom
import time
import json

import os
import sys

sys.dont_write_bytecode = True
sys.path.append(os.path.join('scripts', '..', 'src'))

from codec.core import Encoder, Decoder

def get_filename(path, is_encoding, config):
	
	path, filename = os.path.split(path)
	name, filetype = filename.split('.')

	transfer_type = 'encoded' if is_encoding else 'decoded'
	filetype = config['extension'] if is_encoding else config['decode_format']

	renamed = f'{path}/{transfer_type}-{name}.{filetype}'
	return renamed

def main():
	
	directory = os.getcwd().split('\\')[-1]
	config = json.load(open(('src/' if directory != 'src' else '') + 'config.json', 'r'))
	# print(json.dumps(config))

	print(f'\n==================== [ENCODING] ====================\n')

	# DEMO_IMAGE_PATH = "C:/Users/taaha/Downloads/manifest-OtXaMwL56190865641215613043/QIN LUNG CT/QIN-LSC-0055/07-27-2003-1-CT Thorax wo Contrast-86597/5.000000-THORAX WO  3.0  B41 Soft Tissue-77621/1-016.dcm"
	DEMO_IMAGE_PATH = r"C:\Users\taaha\Downloads\1-55.dcm"
	image = pydicom.read_file(DEMO_IMAGE_PATH).pixel_array

	out_path = 'data/testing.khn'

	encoder = Encoder(config, image, out_path)
	encoder.encode_qoi()
	# encoder.encode_packbits()

	print(f'\n\"QIN LUNG CT/.../{os.path.basename(DEMO_IMAGE_PATH)}\" encoded to \"{out_path}\"')

	print(f'\n==================== [DECODING] ====================\n')

	with open(out_path, 'rb') as encoded:
		file_bytes = encoded.read()

	in_path = get_filename(out_path, False, config)

	decoder = Decoder(config, file_bytes, in_path)
	output = decoder.decode_qoi()
	# output = decoder.decode_packbits()

	print(f'\"{out_path}\" preview decoded to \"{in_path}\"')

	# Confirming that reconstruction error is 0
	error_matrix = image - output
	error = np.count_nonzero(error_matrix)
	print(f'\nReconstruction Error: {error}')

	original_hash = hashlib.sha1(image.tobytes()).hexdigest()
	recovered_hash = hashlib.sha1(output.tobytes()).hexdigest()

	print(f'SHA1 Original Hash:  {original_hash}')
	print(f'SHA1 Recovered Hash: {recovered_hash}')

if __name__ == '__main__':
	main()
