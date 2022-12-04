
# Hybrid DCT-DWT Image compression
### Taaha Khan 2023 Science Fair Project
<br>

## TODO:
- Quadtree image partitioning for dynamic blocking structure
	- If pixel section follows uniform distribution: less compression
	- If pixel section follows unimodal distribution: more compression
- Determine whether DCT or DWT is more efficient for each sector
- Zig-Zag coefficient encoding to bitstream
- Identify optimal color transform
- Compare with public compression datasets

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

### Block Image Partitioning
- [JPEG 2000 Tiling](https://en.wikipedia.org/wiki/JPEG_2000#Tiling)
- [HEVC Encoding](https://www.vcodex.com/hevc-an-introduction-to-high-efficiency-coding/)
