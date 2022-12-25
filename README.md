
# Image Compression
### Taaha Khan 2023 Science Expo Project

## Lossless Encoder

- More than 8 bits per color component (16? 24? 32?) for HDR user market
- Split RGB color into redundant color data (YUV?)
- Consider all (A, B, C) color components as one collective "pixel"

- Read image pixel deltas in redundancy creating order for easier deflation in final compression stage (Raster scan is suboptimal because it jumps from right edge to the left, which can't gurantee significant correlation)

	- Hamiltonian Cycle:
		- PRO: Can retain spacial redundancy through random walks in region
		- CON: Difficult to encode walk order, requires pretransmitted seed
		- CON: Might get only a cut of ROI and move past without exploiting full content redundancy

		MAXIMIZES: cache, near previous, run length

	- Fractal (Ex. Hilberts Curve):
		- PRO: Easily loaded and unloaded, retains some spacial redundancy
		- CON: Quadtree based structure can completely disregard predictably intelligent areas and move away

		MAXIMIZES: cache, near previous, run length
		
	Will these have significant gains from PNG's filter selection which takes into account top and left components together?

	- Optimal Solution?: Traveling Salesmen Problem:
		- Each pixel is a node
		- Each node connected to every other node or local node clusters
		- Weight between nodes is the number of bits required to represent them back-to-back
			Ex. Identical/Similar pixels require fewer bits to encode together, so should be prioritized
		- Find ordering of pixels with minimal total weight using heuristic TSP to maximize redundancy and minimize bit representation
		
		- CONS: Difficult to aid decoder of calculated order in minimal bits

		MAXIMIZES: run length, cache, near previous

	- Burrows-Wheeler Transform:

		- PROS: Maximizes redundancy between neighboring scan orders
		- CONS: Slow if implemented inefficiently (YAY USACO TRAINING!)

		- Move-to-front-Transform:
			- PROS: Minimizes deltas between neighboring pixels in scan order
			- CONS: Doesn't gurantee exact redundancy between pixels

		MAXMIMIZES: cache, near previous, run length
	
- Apply QOI-based condense algorithm (prefer byte alignment with type tags)
	1. Run-length encoding
	2. Recent palatte cache index
	3. Near previous difference encoding
	4. Near previous 2-byte difference encoding
	5. Last case: Full pixel values (Maybe divided by some constant for lower bit representation)

	Potential:
		- Linear difference coefficient between N pixel values
			Ex. (255, 245, 235, 225): {slope = -10, N = 3, start = 255}

- Apply Post-processing compression
	- DEFLATE (LZ77 + Huffman coding) (slow, middle effect, familiar)
	- Arithmetic coding (slow, best effect, unfamiliar)
	- ZStandard Coding (fast, middle effect, familiar)
	- MANIAC (slow, best effect, unfamiliar)

## Compare Compression Ratio, Speed, *Quality
- Public compression datasets
- Lossless: PNG, WebP, FLIF, JPEG2000, QOI
- *Lossy: JPEG, JPEG2000, HEIF

## Important Sources

### Project Structure Notation
- [Simple Documentation](https://github.com/mitcommlab/Coding-Documentation/blob/master/File-Structure-Case-Studies.md#case-study-2-a-simple-hierarchy)

### JPEG Structure
- [JPEG Overview Notebook](https://inst.eecs.berkeley.edu/~ee123/sp16/Sections/JPEG_DCT_Demo.html)
- [Block Quantization](https://en.wikipedia.org/wiki/Quantization_(image_processing))

### Color Transforms
- [YCbCr Wikipedia](https://en.wikipedia.org/wiki/YCbCr)
- [Python Implementation](https://gist.github.com/roytseng-tw/dafc041a65edfdfd86bafcb8129da57d)

### Evaluation Metrics
- [Peak Signal-to-Noise Ratio](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio)
- [Bits Per Pixel (BPP)]

### Block Image Partitioning
- [JPEG 2000 Tiling](https://en.wikipedia.org/wiki/JPEG_2000#Tiling)
- [HEVC Encoding](https://www.vcodex.com/hevc-an-introduction-to-high-efficiency-coding/)
