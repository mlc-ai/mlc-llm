"""Classes denoting multi-modality data used in MLC LLM serving"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple  # noqa: UP035

import tvm
import tvm_ffi
from tvm.runtime import Object, Tensor

from . import _ffi_api


@tvm_ffi.register_object("mlc.serve.Data")
class Data(Object):
    """The base class of multi-modality data (text, tokens, embedding, etc)."""

    def __init__(self):
        pass


@tvm_ffi.register_object("mlc.serve.TextData")
class TextData(Data):
    """The class of text data, containing a text string.

    Parameters
    ----------
    text : str
        The text string.
    """

    def __init__(self, text: str):
        self.__init_handle_by_constructor__(_ffi_api.TextData, text)

    @property
    def text(self) -> str:
        """The text data in `str`."""
        return str(_ffi_api.TextDataGetTextString(self))

    def __str__(self) -> str:
        return self.text


@tvm_ffi.register_object("mlc.serve.TokenData")
class TokenData(Data):
    """The class of token data, containing a list of token ids.

    Parameters
    ----------
    token_ids : List[int]
        The list of token ids.
    """

    def __init__(self, token_ids: List[int]):  # noqa: UP006
        self.__init_handle_by_constructor__(_ffi_api.TokenData, *token_ids)

    @property
    def token_ids(self) -> List[int]:  # noqa: UP006
        """Return the token ids of the TokenData."""
        return list(_ffi_api.TokenDataGetTokenIds(self))


# mypy: disable-error-code="attr-defined"
@tvm_ffi.register_object("mlc.serve.ImageData")
class ImageData(Data):
    """The class of image data, containing the image as Tensor.

    Parameters
    ----------
    image : tvm.runtime.Tensor
        The image data.
    """

    def __init__(self, image: Tensor, embed_size: int):
        self.__init_handle_by_constructor__(_ffi_api.ImageData, image, embed_size)

    @property
    def image(self) -> Tensor:
        """Return the image data."""
        return _ffi_api.ImageDataGetImage(self)

    def __len__(self):
        return _ffi_api.ImageDataGetEmbedSize(self)

    @staticmethod
    def from_url(url: str, config: Dict) -> "ImageData":  # noqa: UP006
        """Get the image from the given URL, process and return the image tensor as TVM Tensor."""

        import base64
        from io import BytesIO

        import numpy as np
        import requests
        from PIL import Image

        if url.startswith("data:image"):
            # The image is encoded in base64 format
            base64_image = url.split(",")[1]
            image_data = base64.b64decode(base64_image)
            image_tensor = Image.open(BytesIO(image_data)).convert("RGB")
        elif url.startswith("http"):
            response = requests.get(url, timeout=5)
            image_tensor = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            raise ValueError(f"Unsupported image URL format: {url}")

        model_type = config["model_type"]
        image_embed_size = ImageData._compute_embed_size(
            model_type, image_tensor.width, image_tensor.height, config
        )
        image_tensor = np.expand_dims(image_tensor, axis=0)  # HWC -> NHWC
        image_features = tvm.runtime.tensor(image_tensor)
        image_data = ImageData(image_features, image_embed_size)
        return image_data

    @staticmethod
    def _compute_embed_size(model_type: str, width: int, height: int, config: Dict) -> int:  # noqa: UP006
        """Compute image embed size based on model type and image dimensions.

        This must match the C++ CalculateResizeShape/CalculateCropShape logic
        in cpp/support/vlm_utils.cc so the embed_size agrees with the actual
        number of tokens produced by image_embed.
        """
        import math  # pylint: disable=import-outside-toplevel

        if model_type == "phi3_v":
            # Replicate C++ CalculateResizeShape (hd_num=4)
            hd_num = 4
            ratio = width / height
            scale = 1
            while scale * math.ceil(scale / ratio) <= hd_num:
                scale += 1
            scale -= 1

            target_w = scale * 336
            target_h = int(target_w / ratio)

            # CalculatePadShape
            pad_h = int(math.ceil(target_h / 336.0) * 336)

            # CalculateCropShape
            crop_h = pad_h // 336
            crop_w = target_w // 336  # == scale

            # Token count formula from phi3v_image.py forward():
            # sub_tokens: h_crop*12 * (w_crop*12+1)  (12=24/2 from 2x2 merge, +1 newline)
            # glb_GN: 1 separator token
            # glb_tokens: 12 * (12 + 1) = 156  (global image with 1x1 crop)
            sub_tokens = crop_h * 12 * (crop_w * 12 + 1)
            glb_tokens = 12 * (12 + 1)
            return sub_tokens + 1 + glb_tokens

        if model_type == "gemma3_v":
            return config["model_config"].get("mm_tokens_per_image", 256)

        if model_type == "qwen3_5_vision":
            # (image_size / patch_size / spatial_merge_size)^2 = (448/16/2)^2 = 196
            return 196

        # Default: (image_size / patch_size)^2
        return ImageData.get_embed_size(config)

    @staticmethod
    def get_embed_size(config: Dict) -> int:  # noqa: UP006
        """Get the image embedding size from the model config file."""
        image_size = config["model_config"]["vision_config"]["image_size"]
        patch_size = config["model_config"]["vision_config"]["patch_size"]
        embed_size = (image_size // patch_size) ** 2
        return embed_size

    @staticmethod
    def get_input_size(config: Dict) -> int:  # noqa: UP006
        """Get the image input size from the model config file."""
        image_size = config["model_config"]["vision_config"]["image_size"]
        return image_size


@dataclass
class SingleRequestStreamOutput:
    """The request stream output of a single request.

    Attributes
    ----------
    delta_token_ids : List[int]
        The new generated tokens since the last callback invocation
        for the input request.

    delta_logprob_json_strs : Optional[List[str]]
        The logprobs JSON strings of the new generated tokens
        since last invocation.

    finish_reason : Optional[str]
        The finish reason of the request when it is finished,
        of None if the request has not finished yet.
    """

    delta_token_ids: List[int]  # noqa: UP006
    delta_logprob_json_strs: Optional[List[str]]  # noqa: UP006
    finish_reason: Optional[str]
    request_final_usage_json_str: Optional[str]
    extra_prefix_string: str


@tvm_ffi.register_object("mlc.serve.RequestStreamOutput")
class RequestStreamOutput(Object):
    """The generated delta request output that is streamed back
    through callback stream function.
    It contains four fields (in order):

    request_id : str
        The id of the request that the function is invoked for.

    stream_outputs : List[SingleRequestStreamOutput]
        The output instances, one for a request.

    Note
    ----
    We do not provide constructor, since in practice only C++ side
    instantiates this class.
    """

    def unpack(self) -> Tuple[str, List[SingleRequestStreamOutput]]:  # noqa: UP006
        """Return the fields of the delta output in a tuple.

        Returns
        -------
        request_id : str
            The id of the request that the function is invoked for.

        stream_outputs : List[SingleRequestStreamOutput]
            The output instances, one for a request.
        """
        fields = _ffi_api.RequestStreamOutputUnpack(self)
        request_final_usage_json_str = fields[4]
        request_id = str(fields[0])
        if request_final_usage_json_str is not None:
            return (
                request_id,
                [SingleRequestStreamOutput([], None, None, request_final_usage_json_str, "")],
            )

        stream_outputs = []
        for i, (delta_token_ids, finish_reason, extra_prefix_string) in enumerate(
            zip(fields[1], fields[3], fields[5])
        ):
            delta_logprob_json_strs = (
                [str(logprob_json_str) for logprob_json_str in fields[2][i]]
                if fields[2] is not None
                else None
            )
            stream_outputs.append(
                SingleRequestStreamOutput(
                    delta_token_ids=list(delta_token_ids),
                    delta_logprob_json_strs=delta_logprob_json_strs,
                    finish_reason=str(finish_reason) if finish_reason is not None else None,
                    request_final_usage_json_str=None,
                    extra_prefix_string=str(extra_prefix_string),
                )
            )
        return request_id, stream_outputs
