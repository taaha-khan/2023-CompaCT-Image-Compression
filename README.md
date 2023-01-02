
# Image Compression
### Taaha Khan 2023 Science Expo Project

## Completed
- Port QOI and Gilbert for python
- Plugin Gilbert to QOI ordering
- Apply ZLib DEFLATE compression to output

## TODO
- Create section skipping algorithm (binary search?)
- Port to HDR byte format
- Get SRGB and HDR testing datasets
- Compare compression ratio, BPP with competitors
- Write up slideshow and abstract

## Lossless Encoder

- More than 8 bits per color component (16? 32?) for [HDR](https://en.wikipedia.org/wiki/Multi-exposure_HDR_capture) user market (medical, photography, security)

	- **32 bits**:
		- 96 bits total per pixel = 12 bytes per pixel
		- Any way to condense N consecutive pixels into less than 12N bytes

- Improved RGB color transform ([YCoCg-R](https://en.wikipedia.org/wiki/YCoCg#The_lifting-based_YCoCg-R_variation)?)
- Consider all (A, B, C) color components as one collective "pixel"

- Read image pixel deltas in redundancy creating order for easier deflation in final compression stage (Raster scan is suboptimal because it jumps from right edge to the left, which can't gurantee significant correlation)

	- **Hamiltonian Cycle:**
		- PRO: Can retain spacial redundancy through random walks in region
		- CON: Difficult to encode walk order, requires pretransmitted seed
		- CON: Might get only a cut of ROI and move past without exploiting full content redundancy

		MAXIMIZES: cache, near previous, run length

		**O(2^N * N^2) good luck**

	- **Fractal** (Ex. Hilberts Curve):

		- PRO: Easily loaded and unloaded, retains some spacial redundancy
		- CON: Quadtree based structure can completely disregard predictably intelligent areas and move away

		MAXIMIZES: cache, near previous, run length

		**Improved Lossless HDR Image Compression Applying Fractal Topological Pixel Restructuring**

		- Each hilbert section is a segment
		- Add segment jumping capabilities with tag in bitstream
		- Optimizes recency cache by jumping over distruptive sections to maintain the as much correlation as possible
		- Later come back and fill in the gaps between jumped segments
		
	Will these have significant gains from PNG's filter selection which takes into account top and left components together?

	- **Optimal Solution?: Traveling Salesmen Problem:**
		- Each pixel is a node
		- Each node connected to every other node or local node clusters
		- Weight between nodes is the number of bits required to represent them back-to-back
			Ex. Identical/Similar pixels require fewer bits to encode together, so should be prioritized
		- Find ordering of pixels with minimal total weight using heuristic TSP to maximize redundancy and minimize bit representation
		
		- CONS: Difficult to aid decoder of calculated order in minimal bits

		MAXIMIZES: run length, cache, near previous

		**Horrendous time complexity and I don't wanna try a million different heuristics**

	- **Burrows-Wheeler Transform:**

		- PROS: Maximizes redundancy between neighboring scan orders
		- CONS: Slow if implemented inefficiently (YAY USACO TRAINING!)

		- **Move-to-front-Transform:**
			- PROS: Minimizes deltas between neighboring pixels in scan order
			- CONS: Doesn't gurantee exact redundancy between pixels

			- Reorder cache with MTF?

		MAXMIMIZES: cache, near previous, run length

		**Every mf already done this**
	
- Apply QOI-based condense algorithm (prefer byte alignment with type tags)
	1. Run-length encoding
	2. Recent palette cache index
	3. Near previous difference encoding
	4. Near previous 2-byte difference encoding
	5. Last case: Full pixel values

	Potential:
		- PNG based alignment

- Apply Post-processing compression
	- DEFLATE (LZ77 + Huffman coding) (slow, middle effect, familiar)
	- Arithmetic coding (slow, best effect, unfamiliar)
	- ZStandard Coding (fast, middle effect, familiar)

## Compare Compression Ratio, Speed, *Quality
- Public compression datasets
- Lossless: PNG, WebP, FLIF, JPEG-LS, JPEG2000, QOI, [HDR](https://en.wikipedia.org/wiki/Category:High_dynamic_range_file_formats)
- *Lossy: JPEG, JPEG2000, HEIF

## Important Sources

### HDR
- [Laws and HDR Usages](researchgate.net/profile/David-Clunie/publication/283356591_What_is_Different_About_Medical_Image_Compression/links/56376a3708aeb786b7044b8a/What-is-Different-About-Medical-Image-Compression.pdf)
- [HDR Image Data](https://unsplash.com/images/stock/hdr)

### Project Structure Notation
- [Simple Documentation](https://github.com/mitcommlab/Coding-Documentation/blob/master/File-Structure-Case-Studies.md#case-study-2-a-simple-hierarchy)

### JPEG Structure
- [JPEG Overview Notebook](https://inst.eecs.berkeley.edu/~ee123/sp16/Sections/JPEG_DCT_Demo.html)
- [Block Quantization](https://en.wikipedia.org/wiki/Quantization_(image_processing))
