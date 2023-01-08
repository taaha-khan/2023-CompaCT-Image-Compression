
import sys
sys.dont_write_bytecode = True

from PIL import Image
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

		if args.file_path.endswith('dcm'):
			image = Image.fromarray(pydicom.imread(args.file_path).pixel_data)
		else:
			try:
				image = Image.open(args.file_path)
			except Exception as exc:
				print(f'Image at \'{args.file_path}\' load failed: {exc}')
				return

		out_path = get_filename(args.file_path, True, config)

		encoder = Encoder(config, image, out_path)
		encoder.encode()

		print(f'{args.file_path} encoded to {out_path}')

	elif args.decode:

		with open(args.file_path, 'rb') as encoded:
			file_bytes = encoded.read()

		out_path = get_filename(args.file_path, False, config)

		decoder = Decoder(config, file_bytes, out_path)
		decoder.decode()

		print(f'{args.file_path} decoded to {out_path}')

if __name__ == '__main__':
	main()
