
import os

encode = 'C:/Users/taaha/Downloads/openjpeg-v2.5.0-windows-x64/openjpeg-v2.5.0-windows-x64/bin/opj_compress.exe'
decode = 'C:/Users/taaha/Downloads/openjpeg-v2.5.0-windows-x64/openjpeg-v2.5.0-windows-x64/bin/opj_decompress.exe'

def png_to_jpeg2000(input_path, output_path):
	encode_command = f'{encode} -i {input_path} -o {output_path}'
	os.system(encode_command)
	return output_path

def jpeg2000_to_png(input_path, output_path):
	decode_command = f'{decode} -i {input_path} -o {output_path}'
	os.system(decode_command)
	return output_path

if __name__ == '__main__':

	input_path = 'data/working/decoded-testing.png'
	output_path = 'data/working/encoded-jpeg2000.jp2'

	png_to_jpeg2000(input_path, output_path)

	raw_size = 512 * 512 * 2
	print(f'raw pixel data size: {raw_size / 1000} KB')

	size = os.path.getsize(output_path)
	print(f'{output_path} size: {size / 1000} KB')

	ratio = raw_size / size
	print(f'JPEG2000-Lossless compression ratio: {ratio:.2f}x')

