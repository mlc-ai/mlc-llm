"""Classes denoting multi-modality data used in MLC LLM serving"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import tvm
import tvm._ffi
from tvm.runtime import Object
from tvm.runtime.ndarray import NDArray

from . import _ffi_api


@tvm._ffi.register_object("mlc.serve.Data")  # pylint: disable=protected-access
class Data(Object):
    """The base class of multi-modality data (text, tokens, embedding, etc)."""

    def __init__(self):
        pass


@tvm._ffi.register_object("mlc.serve.TextData")  # pylint: disable=protected-access
class TextData(Data):
    """The class of text data, containing a text string.

    Parameters
    ----------
    text : str
        The text string.
    """

    def __init__(self, text: str):
        self.__init_handle_by_constructor__(_ffi_api.TextData, text)  # type: ignore  # pylint: disable=no-member

    @property
    def text(self) -> str:
        """The text data in `str`."""
        return str(_ffi_api.TextDataGetTextString(self))  # type: ignore  # pylint: disable=no-member

    def __str__(self) -> str:
        return self.text


@tvm._ffi.register_object("mlc.serve.TokenData")  # type: ignore  # pylint: disable=protected-access
class TokenData(Data):
    """The class of token data, containing a list of token ids.

    Parameters
    ----------
    token_ids : List[int]
        The list of token ids.
    """

    def __init__(self, token_ids: List[int]):
        self.__init_handle_by_constructor__(_ffi_api.TokenData, *token_ids)  # type: ignore  # pylint: disable=no-member

    @property
    def token_ids(self) -> List[int]:
        """Return the token ids of the TokenData."""
        return list(_ffi_api.TokenDataGetTokenIds(self))  # type: ignore  # pylint: disable=no-member


# mypy: disable-error-code="attr-defined"
@tvm._ffi.register_object("mlc.serve.ImageData")  # type: ignore  # pylint: disable=protected-access
class ImageData(Data):
    """The class of image data, containing the image as NDArray.

    Parameters
    ----------
    image : tvm.runtime.NDArray
        The image data.
    """

    def __init__(self, image: NDArray, embed_size: int):
        self.embed_size = embed_size
        self.__init_handle_by_constructor__(_ffi_api.ImageData, image, embed_size)  # type: ignore  # pylint: disable=no-member

    @property
    def image(self) -> NDArray:
        """Return the image data."""
        return _ffi_api.ImageDataGetImage(self)  # type: ignore  # pylint: disable=no-member

    def __len__(self):
        return self.embed_size

    @staticmethod
    def from_url(url: str, config: Dict) -> "ImageData":  # pylint: disable=too-many-locals
        """Get the image from the given URL, process and return the image tensor as TVM NDArray."""

        # pylint: disable=import-outside-toplevel, import-error
        import base64
        from io import BytesIO

        import requests
        from PIL import Image
        from transformers import CLIPImageProcessor

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

        image_input_size = ImageData.get_input_size(config)
        image_embed_size = ImageData.get_embed_size(config)

        image_processor = CLIPImageProcessor(
            size={"shortest_edge": image_input_size},
            crop_size={"height": image_input_size, "width": image_input_size},
        )
        image_features = tvm.nd.array(
            image_processor.preprocess(image_tensor, return_tensors="np")["pixel_values"].astype(
                "float32"
            )
        )
        image_data = ImageData(image_features, image_embed_size)
        return image_data

    @staticmethod
    def get_embed_size(config: Dict) -> int:
        """Get the image embedding size from the model config file."""
        image_size = config["model_config"]["vision_config"]["image_size"]
        patch_size = config["model_config"]["vision_config"]["patch_size"]
        embed_size = (image_size // patch_size) ** 2
        return embed_size

    @staticmethod
    def get_input_size(config: Dict) -> int:
        """Get the image input size from the model config file."""
        image_size = config["model_config"]["vision_config"]["image_size"]
        return image_size

    @staticmethod
    # pylint: disable=too-many-locals,too-many-statements,unused-argument
    def phi3v_from_url(
        url: str, config: Dict
    ) -> "ImageData":  # pylint: disable=too-many-locals, unused-argument
        """Get the image from the given URL, process and return the image tensor as TVM NDArray."""

        def _pad_image(img, hd_num=16):
            # pylint: disable=import-outside-toplevel, import-error
            import numpy as np
            import torchvision

            def padding_336(b):
                _, height = b.size
                tar = int(np.ceil(height / 336) * 336)
                top_padding = int((tar - height) / 2)
                bottom_padding = tar - height - top_padding
                left_padding = 0
                right_padding = 0
                b = torchvision.transforms.functional.pad(
                    b,
                    [left_padding, top_padding, right_padding, bottom_padding],
                    fill=[255, 255, 255],
                )
                return b

            width, height = img.size
            trans = False
            if width < height:
                # pylint: disable=no-member
                img = img.transpose(Image.TRANSPOSE)
                trans = True
                width, height = img.size
            ratio = width / height
            scale = 1
            while scale * np.ceil(scale / ratio) <= hd_num:
                scale += 1
            scale -= 1
            new_w = int(scale * 336)
            new_h = int(new_w / ratio)
            img = torchvision.transforms.functional.resize(
                img,
                [new_h, new_w],
            )
            img = padding_336(img)
            width, height = img.size
            if trans:
                img = img.transpose(Image.TRANSPOSE)
            return img

        def _pad_to_max_num_crops_tensor(images, max_crops=5):
            """
            images: B x 3 x H x W, B<=max_crops
            """
            b, _, h, w = images.shape
            if b < max_crops:
                pad = torch.zeros(max_crops - b, 3, h, w, dtype=images.dtype, device=images.device)
                images = torch.cat([images, pad], dim=0)
            return images

        # pylint: disable=import-outside-toplevel, import-error
        import base64
        from io import BytesIO

        import requests
        import torch
        import torchvision
        from PIL import Image

        num_crops = 16

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

        image_mean = [0.48145466, 0.4578275, 0.40821073]
        image_std = [0.26862954, 0.26130258, 0.27577711]
        img_processor = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(image_mean, image_std),
            ]
        )

        image_tensor = _pad_image(image_tensor, 16)
        image_tensor = img_processor(image_tensor)  # from IPL image to torch tensor

        # resize to 336x336x3 global image
        global_image = torch.nn.functional.interpolate(
            image_tensor.unsqueeze(0).float(),
            size=(336, 336),
            mode="bicubic",
        ).to(image_tensor.dtype)

        # [(3, h, w)], where h, w is multiple of 336
        h = image_tensor.size(1)
        w = image_tensor.size(2)
        hd_images_reshape = (
            image_tensor.reshape(1, 3, h // 336, 336, w // 336, 336)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(-1, 3, 336, 336)
            .contiguous()
        )
        # concat global image and local image
        hd_images_reshape = torch.cat([global_image] + [hd_images_reshape], dim=0)

        image_transformed = [_pad_to_max_num_crops_tensor(hd_images_reshape, num_crops + 1)]
        image_transformed = torch.stack(image_transformed, dim=0)

        padded_images = image_transformed

        image_features = tvm.nd.array(
            padded_images.cpu().numpy(),
        )
        image_data = ImageData(image_features, 1024)
        return image_data


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

    delta_token_ids: List[int]
    delta_logprob_json_strs: Optional[List[str]]
    finish_reason: Optional[str]
    request_final_usage_json_str: Optional[str]
    extra_prefix_string: str


@tvm._ffi.register_object("mlc.serve.RequestStreamOutput")  # pylint: disable=protected-access
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

    def unpack(self) -> Tuple[str, List[SingleRequestStreamOutput]]:
        """Return the fields of the delta output in a tuple.

        Returns
        -------
        request_id : str
            The id of the request that the function is invoked for.

        stream_outputs : List[SingleRequestStreamOutput]
            The output instances, one for a request.
        """
        fields = _ffi_api.RequestStreamOutputUnpack(self)  # type: ignore  # pylint: disable=no-member
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
