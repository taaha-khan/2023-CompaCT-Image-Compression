
# Image Compression
### Taaha Khan 2023 Science Expo Project

## Title Ideas

**Keywords**
- Improved/Applying/With/Through/Via
- Lossless, Perfect Quality
- HDR (High Dynamic Range), High (Bit/Color) Depth
- Image Compression
- CT (Computed Tomography) Scan
- Pixel
- Fractal
- Topological
- Reordering/Restructering/Transform
- Heuristic Hierarchical Clustering/Segmentation

**High Color Depth CT Scan Compression Applying Heuristic Hierarchical Clustering and Fractal Topological Pixel Reordering**

**Heuristic Fractal Pixel Segmentation for High Color Depth CT Scan Compression**

## Completed
- Port QOI and Gilbert for python
- Plugin Gilbert to QOI ordering
- Apply ZLib DEFLATE compression to output
- Get SRGB and HDR testing datasets
- Hirerichal Clustering: Create section skipping algorithm (binary search?)

## Ideas
- TILE COMPRESSION: Group pixels into (hilbert cluster?) sections and encode as single cluster
- ZIPPER: Read image left edge, right edge, etc. in zipper format for vertical symmetry
	- PROBLEM: Not perfect vertical symmetry, very bad offset error

- FRACTAL: 
	- Take NxN segments of image
	- Look at neighboring segments
	- Group segments if meshing is better

	EXAMPLE:

	let BlockA
	let BlockB = BlockA.move(DOWN_3, RIGHT_4)

	mesh = zip(BlockA, BlockB)
	if AvgAbsDiff(mesh) < AvgAbsDiff(BlockA) + AvgAbsDiff(BlockB):
		# mesh pixels from BlockA and BlockB

## TODO
- Finish Cluster plugin
- Compare compression ratio, BPP with competitors
- Write up slideshow and abstract

## Lossless Encoder

- More than 8 bits per color component (16) for [HDR](https://en.wikipedia.org/wiki/Multi-exposure_HDR_capture) user market (medical)

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
	
- Apply QOI or PackBits based condense algorithm (prefer byte alignment)
	1. Run-length encoding
	2. Recent palette cache index
	3. Near previous difference encoding
	4. Near previous 2-byte difference encoding
	5. Last case: Full pixel values

- Apply Post-processing DEFLATE (LZ77 + Huffman coding)

## Compare Compression Ratio, Speed
- Lossless: PNG, WebP, FLIF, JPEG-LS, JPEG2000, QOI, [HDR](https://en.wikipedia.org/wiki/Category:High_dynamic_range_file_formats)
- DICOM ALGORITHM: https://pydicom.github.io/pydicom/dev/tutorials/pixel_data/compressing.html#compressing-using-pydicom

## Important Sources

- [Unavailable File Formats](https://en.wikipedia.org/wiki/List_of_file_formats)

### Lossless Compression
- [QOI Blog](https://phoboslab.org/log/2021/11/qoi-fast-lossless-image-compression)
- [QOI Source](https://github.com/phoboslab/qoi)
- [QOI Python Port](https://github.com/mathpn/py-qoi)
- [QOI Analysis](https://wiesmann.codiferes.net/wordpress/archives/33156)
- [WEBP Spec](https://developers.google.com/speed/webp/docs/webp_lossless_bitstream_specification)
- [PNG YouTube](https://www.youtube.com/watch?v=EFUYNoFRHQI)
- [JPEG2000-Lossless Documentation](https://github.com/uclouvain/openjpeg/wiki/DocJ2KCodec)

### DICOM CT Dataset
- [GE, non-equistant, gantry/detector tilt](https://www.aliza-dicom-viewer.com/download/datasets)
- **[QIN LUNG CT Dataset](https://wiki.cancerimagingarchive.net/display/Public/QIN+LUNG+CT#19039647a520d4e15ee04e84bf26ec185e5403b7)**
- [Laws and HDR Usages](researchgate.net/profile/David-Clunie/publication/283356591_What_is_Different_About_Medical_Image_Compression/links/56376a3708aeb786b7044b8a/What-is-Different-About-Medical-Image-Compression.pdf)
- [DICOM Standard](https://dicom.nema.org/medical/Dicom/2016e/output/chtml/part05/sect_8.2.html)

### Autoencoder
- [Autoencoder Demo](https://www.datacamp.com/tutorial/reconstructing-brain-images-deep-learning)

### Project Structure Notation
- [Simple Documentation](https://github.com/mitcommlab/Coding-Documentation/blob/master/File-Structure-Case-Studies.md#case-study-2-a-simple-hierarchy)

## Licensing (Closed Source Project)
- https://creativecommons.org/licenses/by-nc-nd/3.0/
- https://creativecommons.org/licenses/by-nc-nd/3.0/legalcode