
import sys
sys.dont_write_bytecode = True

import time
import json
import os

# from codec.core import Encoder, Decoder

if __name__ == '__main__':

	directory = os.getcwd().split('\\')[-1]
	config = json.load(open(('src/' if directory != 'src' else '') + 'config.json', 'r'))
	print(json.dumps(config))

	# e = Encoder(None, config)
	# e.encode()

	# print(e)

	# d = Decoder(config)
	# d.decode()

	# print(d)
	
	size = os.path.getsize('C:\TTNR\projects\ISEF\ISEF-2023\data\examples\horses.png')
	print('original', size, 'bytes')

	size = os.path.getsize('C:\TTNR\projects\ISEF\ISEF-2023\data\examples\horses.qoi')
	print('compressed', size, 'bytes')
	
