# qwen2_vl_image.py
# Contains image preprocessing, ViT definition, and other image-related operations.

from typing import List, Tuple

from tvm import relax, te, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_llm.model.vision import ImageProcessor


class QWen2VLImagePreprocessor(nn.Module):
    def __init__(
        self,
        do_resize: bool = True,
        resample: str = "bicubic",
        do_rescale: bool = True,
        rescale_factor: float = 1/255.0,
        do_normalize: bool = True,
        image_mean: Tensor = OPENAI_CLIP_MEAN,
        image_std: Tensor = OPENAI_CLIP_STD,
        min_pixels: int = 56*56,
        max_pixels: int = 28*28*1280,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
    ):
        self.do_resize = do_resize
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.image_processor = ImageProcessor()

    def smart_resize(height: int, width: int, factor: int=28, min_pixels: int = 56*56, max_pixels: int = 14*14*4*1280) -> Tuple[int, int]:
        """
        Rescales the image, similar to the Huggingface implementation, so that the following conditions are met:

        1. Both dimensions (height and width) are divisible by 'factor'.
        2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
        3. The aspect ratio of the image is maintained as closely as possible.
        """
        
        if height < factor or width < factor:
            raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
        elif max(height, width) / min(height, width) > 200:
            raise ValueError(
                f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
            )
        h_bar = tir.round(height / factor) * factor
        w_bar = tir.round(width / factor) * factor
        if h_bar * w_bar > max_pixels:
            beta = tir.sqrt((height * width) / max_pixels)
            h_bar = tir.floor(height / beta / factor) * factor
            w_bar = tir.floor(width / beta / factor) * factor
        elif h_bar * w_bar < min_pixels:
            beta = tir.sqrt(min_pixels / (height * width))
            h_bar = tir.ceil(height * beta / factor) * factor
            w_bar = tir.ceil(width * beta / factor) * factor
        return h_bar, w_bar

    def forward(self, pixel_values: Tensor, resized_height, resized_width) -> Tensor:
        pixel_values = op.permute_dims(pixel_values, axes=(0, 3, 1, 2)) # NHWC -> NCHW
        if self.do_resize:
            hbar, wbar = self.smart_resize(pixel_values.shape[2], pixel_values.shape[3], factor=self.patch_size*self.merge_size)
            pixel_values = self.image_processor.resize(pixel_values, params={"height": hbar, "width": wbar})
        if self.do_rescale:
            pixel_values = self.image_processor.rescale(pixel_values, factor=self.rescale_factor)
        if self.do_normalize:
            pixel_values = self.image_processor.normalize(pixel_values, mean=self.image_mean, std=self.image_std)
        
        # TODO no padding in HF but do we need?
        return pixel_values

class QWen2VLVisionTransformer:
    # TODO Not CLIP, uses original ViT (CLIP also bases on this)
    def __init__(self):
        pass

