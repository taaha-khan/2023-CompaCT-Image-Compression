
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

	# input_path = "C:/Users/taaha/Downloads/manifest-OtXaMwL56190865641215613043/QIN LUNG CT/QIN-LSC-0055/07-27-2003-1-CT Thorax wo Contrast-86597/5.000000-THORAX WO  3.0  B41 Soft Tissue-77621/1-016.dcm"
	input_path = r"C:\Users\taaha\Downloads\1-55.dcm"
	image = pydicom.read_file(input_path).pixel_array

	encoded_path = f'data/working/testing.{config["extension"]}'

	start_encode = time.process_time()

	encoder = Encoder(config, image, encoded_path)
	encoder.encode()

	elapsed_encode = time.process_time() - start_encode
	print(f'\nEncoding Elapsed Time: {elapsed_encode:.2f} sec')

	print(f'\n\"QIN LUNG CT/.../{os.path.basename(input_path)}\" encoded to \"{encoded_path}\"')

	print(f'\n==================== [DECODING] ====================\n')

	with open(encoded_path, 'rb') as encoded:
		file_bytes = encoded.read()

	decoded_path = get_filename(encoded_path, False, config)

	start_decode = time.process_time()

	decoder = Decoder(config, file_bytes, decoded_path)
	output = decoder.decode()

	elapsed_decode = time.process_time() - start_decode
	print(f'Decoding Elapsed Time: {elapsed_decode:.2f} sec')

	print(f'\n\"{encoded_path}\" preview decoded to \"{decoded_path}\"')

	# Confirming that reconstruction error is 0
	error_matrix = image - output
	error = np.count_nonzero(error_matrix)
	print(f'\nReconstruction Error: {error}')

	# Confirming that SHA hashes are the same
	original_hash = hashlib.sha1(image.tobytes()).hexdigest()
	recovered_hash = hashlib.sha1(output.tobytes()).hexdigest()

	print(f'SHA1 Original Hash:  {original_hash}')
	print(f'SHA1 Recovered Hash: {recovered_hash}')

if __name__ == '__main__':
	main()
