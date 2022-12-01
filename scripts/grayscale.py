
from PIL import Image

import sys
import os

def main(filepath: str) -> None:
	
	img = Image.open(filepath).convert('L')

	path, filename = os.path.split(filepath)
	name, filetype = filename.split('.')

	renamed = f'{path}/{name}-gray.{filetype}'
	img.save(renamed)

	print(f'\'{filepath}\' converted to grayscale and saved to \'{renamed}\'')

if __name__ == '__main__':
	main(sys.argv[1])