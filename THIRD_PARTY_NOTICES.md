# Third-Party Licenses and Attribution

**AZZANI Image Enhancements** (`azzani_enhance`) is released under the **MIT License** (see `LICENSE`).
The software integrates and downloads open-source Python packages and permissive-license pretrained model weights on demand.

## Software Libraries

| Component | License | Purpose |
| :--- | :--- | :--- |
| [Spandrel](https://github.com/chaiNNer-org/spandrel) | MIT | Unified PyTorch image restoration & super-resolution model loader |
| [PyTorch](https://pytorch.org/) / TorchVision | BSD-3-Clause | Tensor execution and CUDA acceleration |
| [OpenCV (`opencv-python-headless`)](https://opencv.org/) | Apache-2.0 | Color space math, resampling, geometric transforms, YuNet DNN inference |
| [Pillow](https://python-pillow.org/) | HPND / MIT-CMU | Image I/O, EXIF orientation, ICC color profile management |
| [NumPy](https://numpy.org/) & [SciPy](https://scipy.org/) | BSD-3-Clause | Float32 signal processing and statistical analysis |
| [Safetensors](https://github.com/huggingface/safetensors) | Apache-2.0 | Safe zero-pickle tensor weight deserialization |
| [Rich](https://github.com/Textualize/rich) | MIT | Terminal progress bars and structured diagnostic output |
| [Hugging Face Hub](https://github.com/huggingface/huggingface_hub) | Apache-2.0 | Model metadata and resilient file download |

## Default Pretrained Model Weights (Downloaded On Demand)

Models are **not** bundled inside the repository; they are fetched on first use with cryptographic SHA-256 verification and remain under their respective authors' permissive licenses:

| Model ID | Author / Origin | License | Attribution |
| :--- | :--- | :--- | :--- |
| `1xDeJPG_realplksr_otf` | Philip Hofmann (`Phips`) | CC-BY-4.0 | https://huggingface.co/Phips/1xDeJPG_realplksr_otf |
| `1xDeNoise_realplksr_otf` | Philip Hofmann (`Phips`) | CC-BY-4.0 | https://huggingface.co/Phips/1xDeNoise_realplksr_otf |
| `1xExposureCorrection_compact` | Philip Hofmann (`Phips`) | CC-BY-4.0 | https://huggingface.co/Phips/1xExposureCorrection_compact |
| `NAFNet-GoPro-width32` | Megvii Research (`NAFNet`) | MIT | https://github.com/megvii-research/NAFNet |
| `2xNomosUni_span_multijpg_ldl` | Philip Hofmann (`Phips`) | CC-BY-4.0 | https://huggingface.co/Phips/2xNomosUni_span_multijpg_ldl |
| `2xHFA2k_LUDVAE_SPAN` | Philip Hofmann (`Phips`) | CC-BY-4.0 | https://huggingface.co/Phips/2xHFA2k_LUDVAE_SPAN |
| `4xNomosWebPhoto_RealPLKSR` | Philip Hofmann (`Phips`) | CC-BY-4.0 | https://huggingface.co/Phips/4xNomosWebPhoto_RealPLKSR |
| `4xRealWebPhoto_v4_dat2` | Philip Hofmann (`Phips`) | CC-BY-4.0 | https://huggingface.co/Phips/4xRealWebPhoto_v4_dat2 |
| `4xNomos2_hq_dat2` | Philip Hofmann (`Phips`) | CC-BY-4.0 | https://huggingface.co/Phips/4xNomos2_hq_dat2 |
| `4xFaceUpDAT` | Philip Hofmann (`Phips`) | CC-BY-4.0 | https://huggingface.co/Phips/4xFaceUpDAT |
| `4xHFA2k_ludvae_realplksr_dysample` | Philip Hofmann (`Phips`) | CC-BY-4.0 | https://huggingface.co/Phips/4xHFA2k_ludvae_realplksr_dysample |
| `GFPGANv1.4` | Tencent ARC | Apache-2.0 | https://github.com/TencentARC/GFPGAN |
| `YuNet (face_detection_yunet_2023mar)` | OpenCV Zoo / Shiqi Yu | MIT | https://github.com/opencv/opencv_zoo |
| `SFace (face_recognition_sface_2021dec)` | OpenCV Zoo / Zhong et al. | Apache-2.0 | https://github.com/opencv/opencv_zoo |

> **Note:** `spandrel_extra_arches` is intentionally **not** required by default so that all default dependencies and model weights remain commercially permissive under MIT / Apache-2.0 / BSD-3 / CC-BY-4.0. Users who pass custom local checkpoints with extra architectures may optionally install `spandrel_extra_arches` in their environment.
