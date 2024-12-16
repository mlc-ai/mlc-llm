"""
Implementation for Phi architecture.
"""

from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Module, Tensor
from tvm.script import tir as T

from mlc_llm.model.vision import CLIPVisionModel
from mlc_llm.support.config import ConfigBase


# mypy: disable-error-code="attr-defined"
# pylint: disable=invalid-name,missing-docstring
class ImageProjection(Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: ConfigBase):
        super().__init__()
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size * 4, config.hidden_size, bias=True
        )
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(self, image_features: Tensor) -> Tensor:
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class Phi3ImageEmbedding(Module):
    def __init__(self, config: ConfigBase):
        super().__init__()

        self.img_processor = CLIPVisionModel(config.vision_config)
        self.image_dim_out = config.img_processor["image_dim_out"]

        self.glb_GN = nn.Parameter((1, 1, self.image_dim_out * 4))
        self.sub_GN = nn.Parameter((1, 1, 1, self.image_dim_out * 4))

        self.img_projection = ImageProjection(config)
        self.image_size = config.vision_config.image_size

    def get_img_features(self, img_embeds: Tensor) -> Tensor:
        img_processor_output = self.img_processor(img_embeds)
        patch_feature = nn.op.split(img_processor_output, indices_or_sections=[1], axis=1)
        return patch_feature[1]

    # pylint: disable=too-many-locals,too-many-locals,unused-argument
    def forward(self, pixel_values: Tensor, raw_image_h, raw_image_w) -> Tensor:
        h = 3  # raw_image_h // self.image_size
        w = 4  # raw_image_w // self.image_size
        B_ = h * w
        C = self.image_dim_out

        # img_embeds = nn.op.squeeze(pixel_values, 0)
        img_features = self.get_img_features(pixel_values)
        H = T.int32((img_features.shape[1] ** 0.5))
        img_features = nn.op.split(img_features, indices_or_sections=[1], axis=0)
        global_img_feature = img_features[0]
        global_img_feature = nn.op.reshape(global_img_feature, ([1, H, H, C]))
        global_img_feature = nn.op.reshape(global_img_feature, ([1, H // 2, 2, H // 2, 2, C]))
        global_img_feature = nn.op.permute_dims(global_img_feature, axes=([0, 1, 3, 2, 4, 5]))
        glb_img = nn.op.reshape(global_img_feature, ([1, H // 2, H // 2, 4 * C]))

        temp_glb_GN = nn.op.repeat(self.sub_GN, int(H // 2), 1)
        glb_img = nn.op.concat([glb_img, temp_glb_GN], dim=2)
        glb_img = nn.op.reshape(glb_img, ([1, -1, 4 * C]))

        sub_img = img_features[1]
        sub_img = nn.op.split(sub_img, indices_or_sections=[12], axis=0)
        sub_img = sub_img[0]
        sub_img = nn.op.reshape(sub_img, ([B_, H, H, C]))
        sub_img = nn.op.reshape(sub_img, ([B_, H // 2, 2, H // 2, 2, C]))
        sub_img = nn.op.permute_dims(sub_img, axes=([0, 1, 3, 2, 4, 5]))
        sub_img = nn.op.reshape(sub_img, ([B_, H // 2, 2, H // 2, 2, C]))
        sub_img = nn.op.reshape(sub_img, ([B_, -1, 4 * C]))
        sub_img = nn.op.reshape(sub_img, ([1, h, w, 12, 12, -1]))
        sub_img = nn.op.permute_dims(sub_img, axes=([0, 1, 3, 2, 4, 5]))
        sub_img = nn.op.reshape(sub_img, ([1, h * 12, w * 12, 4 * C]))

        temp_sub_GN = nn.op.repeat(self.sub_GN, h * 12, 1)
        sub_img = nn.op.concat([sub_img, temp_sub_GN], dim=2)
        sub_img = nn.op.reshape(sub_img, ([1, -1, 4 * C]))

        output_img = nn.op.concat([sub_img, self.glb_GN, glb_img], dim=1)

        img_set_tensor = self.img_projection(output_img)
        img_set_tensor = nn.op.squeeze(img_set_tensor, 0)
        return img_set_tensor
