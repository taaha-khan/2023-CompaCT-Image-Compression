
import sys
sys.dont_write_bytecode = True

from PIL import Image
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
	parser.add_argument('-f', '--file-path', type = str, required = True)
	parser.add_argument('-o', '--out-path', type = str, required = False)
	args = parser.parse_args()

	directory = os.getcwd().split('\\')[-1]
	config = json.load(open(('src/' if directory != 'src' else '') + 'config.json', 'r'))
	# print(json.dumps(config))

	if args.encode:

		"""
		if args.file_path.endswith('dcm'):
			# image = Image.fromarray(pydicom.read_file(args.file_path).pixel_array)
			image = pydicom.read_file(args.file_path).pixel_array
		else:
			try:
				image = Image.open(args.file_path)
			except Exception as exc:
				print(f'Image at \'{args.file_path}\' load failed: {exc}')
				return
		"""

		path = "C:/Users/taaha/Downloads/manifest-OtXaMwL56190865641215613043/QIN LUNG CT/QIN-LSC-0055/07-27-2003-1-CT Thorax wo Contrast-86597/5.000000-THORAX WO  3.0  B41 Soft Tissue-77621/1-016.dcm"

		image = pydicom.read_file(path).pixel_array

		# out_path = get_filename(args.file_path, True, config)
		out_path = 'data/encoded_temp.khn'

		encoder = Encoder(config, image, out_path)
		encoder.encode_packbits()
		# encoder.encode_qoi()

		# print(f'\"{path}\" encoded to \"{out_path}\"')
		# print(f'{args.file_path} encoded to {out_path}')

	elif args.decode:

		with open(args.file_path, 'rb') as encoded:
			file_bytes = encoded.read()

		out_path = get_filename(args.file_path, False, config)

		decoder = Decoder(config, file_bytes, out_path)
		decoder.decode_packbits()

		print(f'{args.file_path} decoded to {out_path}')

if __name__ == '__main__':
	main()
