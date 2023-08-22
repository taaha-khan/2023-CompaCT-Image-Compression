
## CompaCT: Fractal-Based Heuristic Pixel Segmentation for Lossless Compression of High-Color DICOM Medical Images

[![DOI](https://zenodo.org/badge/572831257.svg)](https://zenodo.org/badge/latestdoi/572831257)

### Taaha Khan
### taahajkhan@gmail.com

**Abstract.** Medical image compression is a widely studied field of data processing due to its prevalence in modern digital databases. These domains require a high color depth of 12 bits per pixel component for accurate analysis by physicians, primarily in the DICOM format. Standard raster-based compression of images via filtering and entropy coding is well-known; however, it remains suboptimal in the medical domain due to non-specialized implementations. This study proposes a lossless medical image compression algorithm, CompaCT, that aims to target spatial features and patterns of pixel concentration for dynamically enhanced data processing. The algorithm employs fractal pixel traversal coupled with a novel approach of segmentation and meshing between pixel blocks for preprocessing. Furthermore, delta and entropy coding are applied to this domain for a complete compression pipeline. The proposal demonstrates that the data compression achieved via fractal segmentation preprocessing yields enhanced image compression results while remaining lossless in its reconstruction accuracy. CompaCT is evaluated in its compression ratios on 3954 high-color CT scans against the efficiency of industry-standard compression techniques (i.e., JPEG2000, RLE, ZIP, PNG). Its reconstruction performance is assessed with error metrics to verify lossless image recovery after decompression. The results demonstrate that CompaCT can compress and losslessly reconstruct medical images, being 37% more space-efficient than industry-standard compression systems.

**Keywords:** Compression, DICOM, Heuristic, Fractal, Segmentation.
