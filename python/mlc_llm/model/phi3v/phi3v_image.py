"""
Implementation for Phi architecture.
"""

from tvm import relax, tirx
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Module, Tensor, op
from tvm.script import s_tir as Ts
from tvm.script import tirx as T

from mlc_llm.model.vision import CLIPVisionModel
from mlc_llm.support.config import ConfigBase


# mypy: disable-error-code="attr-defined"
class ImageProjection(Module):
    def __init__(self, config: ConfigBase):
        super().__init__()
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size * 4, config.hidden_size, bias=True
        )
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(self, image_features: Tensor) -> Tensor:
        shape_1 = tirx.Var("shape_1", "int64")
        image_features = op.wrap_nested(
            relax.BlockBuilder()
            .current()
            .match_cast(
                image_features._expr,
                relax.TensorType([shape_1, image_features.shape[1]], image_features.dtype),
            ),
            "image_features",
        )

        hidden_states = self.linear_1(image_features)

        shape_2 = tirx.Var("shape_2", "int64")
        hidden_states = op.wrap_nested(
            relax.BlockBuilder()
            .current()
            .match_cast(
                hidden_states._expr,
                relax.TensorType([shape_2, hidden_states.shape[1]], hidden_states.dtype),
            ),
            "hidden_states",
        )

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

    def apply_schedule(self, sch, block, bdx=32, tile=[32, 32]):
        loop_x, loop_y = sch.get_loops(block)[-2:]
        xo, xi = sch.split(loop_x, factors=[tile[0], None])
        yo, yi = sch.split(loop_y, factors=[tile[1], None])
        sch.reorder(xo, yo, xi, yi)
        t = sch.fuse(xo, yo)
        ty, tx = sch.split(t, factors=[None, bdx])
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")

    def dyn_repeat_4d_tensor(self, input_tensor, r0, r1, r2, r3) -> Tensor:
        assert 4 == input_tensor.ndim, "input_tensor should be 4D data tensor"

        def create_dyn_repeat_func(dtype):
            n = T.dynamic("n", "int64")
            c = T.dynamic("c", "int64")
            h = T.dynamic("h", "int64")
            w = T.dynamic("w", "int64")
            ch0 = T.dynamic("ch0", "int64")
            ch1 = T.dynamic("ch1", "int64")
            ch2 = T.dynamic("ch2", "int64")
            ch3 = T.dynamic("ch3", "int64")

            @Ts.prim_func
            def dyn_repeat_4d_tensor_func(  # pylint disable=too-many-locals
                input_tensor_buf: T.Tensor((n, c, h, w), dtype),
                ch0: T.int64(),
                ch1: T.int64(),
                ch2: T.int64(),
                ch3: T.int64(),
                out_buf: T.Tensor((n * ch0, c * ch1, h * ch2, w * ch3), dtype),
            ):
                T.func_attr({"op_pattern": 8, "tirx.noalias": True, "tirx.is_scheduled": 1})

                for n_idx in T.thread_binding(n * ch0, thread="blockIdx.x"):
                    for c_idx in T.thread_binding(c * ch1, thread="blockIdx.y"):
                        for h_idx, w_idx in T.grid(h * ch2, w * ch3):
                            with Ts.sblock("dyn_repeat_4d_tensor"):
                                Ts.reads(input_tensor_buf[n_idx, c_idx, h_idx, w_idx])
                                Ts.writes(out_buf[n_idx, c_idx, h_idx, w_idx])
                                out_buf[n_idx, c_idx, h_idx, w_idx] = input_tensor_buf[
                                    n_idx % n, c_idx % c, h_idx % h, w_idx % w
                                ]

            return dyn_repeat_4d_tensor_func

        n, c, h, w = input_tensor.shape
        out = op.tensor_ir_op(
            create_dyn_repeat_func(input_tensor.dtype),
            "dyn_repeat_4d_tensor",
            [input_tensor, r0, r1, r2, r3],
            [Tensor.placeholder([n * r0, c * r1, h * r2, w * r3], input_tensor.dtype)],
        )
        return out

    def dyn_concate_dim_2(self, input_1, input_2) -> Tensor:
        def create_dyn_concate_func(dtype):
            n = T.dynamic("n", "int64")
            c = T.dynamic("c", "int64")
            h1 = T.dynamic("h1", "int64")
            h2 = T.dynamic("h2", "int64")
            w = T.dynamic("w", "int64")

            @Ts.prim_func
            def dyn_concate_dim_2_func(
                input_1_buf: T.Tensor((n, c, h1, w), dtype),
                input_2_buf: T.Tensor((n, c, h2, w), dtype),
                out_buf: T.Tensor((n, c, h1 + h2, w), dtype),
            ):
                T.func_attr({"op_pattern": 8, "tirx.noalias": True, "tirx.is_scheduled": 1})

                for n_idx in T.thread_binding(n, thread="blockIdx.x"):
                    for c_idx in T.thread_binding(c, thread="blockIdx.y"):
                        for h_idx, w_idx in T.grid(h1 + h2, w):
                            with Ts.sblock("dyn_concate_dim_2"):
                                Ts.reads(input_1_buf[n_idx, c_idx, h_idx, w_idx])
                                Ts.writes(out_buf[n_idx, c_idx, h_idx, w_idx])
                                if h_idx < h1:
                                    out_buf[n_idx, c_idx, h_idx, w_idx] = input_1_buf[
                                        n_idx, c_idx, h_idx, w_idx
                                    ]
                                else:
                                    out_buf[n_idx, c_idx, h_idx, w_idx] = input_2_buf[
                                        n_idx, c_idx, h_idx - h1, w_idx
                                    ]

            return dyn_concate_dim_2_func

        n1, c1, h1, w1 = input_1.shape
        n2, c2, h2, w2 = input_2.shape
        assert n1 == n2 and c1 == c2 and w1 == w2

        out = op.tensor_ir_op(
            create_dyn_concate_func(input_1.dtype),
            "dyn_concate_dim_2",
            [input_1, input_2],
            [Tensor.placeholder([n1, c1, h1 + h2, w1], input_1.dtype)],
        )
        return out

    def dyn_concate_dim_1(self, input_1, input_2) -> Tensor:
        def create_dyn_concate_func(dtype):
            c = T.dynamic("c", "int64")
            h1 = T.dynamic("h1", "int64")
            h2 = T.dynamic("h2", "int64")
            w = T.dynamic("w", "int64")

            @Ts.prim_func
            def dyn_concate_dim_1_func(
                input_1_buf: T.Tensor((c, h1, w), dtype),
                input_2_buf: T.Tensor((c, h2, w), dtype),
                out_buf: T.Tensor((c, h1 + h2, w), dtype),
            ):
                T.func_attr({"op_pattern": 8, "tirx.noalias": True, "tirx.is_scheduled": 1})

                for c_idx in T.thread_binding(c, thread="blockIdx.y"):
                    for h_idx, w_idx in T.grid(h1 + h2, w):
                        with Ts.sblock("dyn_concate_dim_1"):
                            Ts.reads(input_1_buf[c_idx, h_idx, w_idx])
                            Ts.writes(out_buf[c_idx, h_idx, w_idx])
                            if h_idx < h1:
                                out_buf[c_idx, h_idx, w_idx] = input_1_buf[c_idx, h_idx, w_idx]
                            else:
                                out_buf[c_idx, h_idx, w_idx] = input_2_buf[c_idx, h_idx - h1, w_idx]

            return dyn_concate_dim_1_func

        c1, h1, w1 = input_1.shape
        c2, h2, w2 = input_2.shape
        assert c1 == c2 and w1 == w2

        out = op.tensor_ir_op(
            create_dyn_concate_func(input_1.dtype),
            "dyn_concate",
            [input_1, input_2],
            [Tensor.placeholder([c1, h1 + h2, w1], input_1.dtype)],
        )
        return out

    def get_img_features(self, img_embeds: Tensor) -> Tensor:
        img_processor_output = self.img_processor(img_embeds)
        patch_feature = nn.op.split(img_processor_output, indices_or_sections=[1], axis=1)
        return patch_feature[1]

    def reshape_hd_patches_2x2merge(self, image_features, h_crop, w_crop):
        N, L, C = image_features.shape
        num_images = 1
        H = int(L**0.5)
        image_features = nn.op.reshape(image_features, ([N, H, H, C]))  # N, 24, 24, 1024
        image_features = nn.op.reshape(
            image_features, ([N, H // 2, 2, H // 2, 2, C])
        )  # N, 12, 2, 12, 2, 1024

        new_s1 = tirx.Var("new_s1", "int64")
        new_s2 = tirx.Var("new_s2", "int64")

        image_features = op.wrap_nested(
            relax.BlockBuilder()
            .current()
            .match_cast(
                image_features._expr,
                relax.TensorType(
                    [
                        image_features.shape[0],
                        new_s1,
                        image_features.shape[2],
                        new_s2,
                        image_features.shape[4],
                        image_features.shape[5],
                    ],
                    image_features.dtype,
                ),
            ),
            "image_features_1",
        )

        image_features = nn.op.permute_dims(
            image_features, axes=([0, 1, 3, 2, 4, 5])
        )  # N, 12, 12, 2, 2, 1024
        image_features = nn.op.reshape(image_features, ([N, -1, 4 * C]))  # N, 144, 4096
        image_features = nn.op.reshape(
            image_features, ([num_images, h_crop, w_crop, H // 2, H // 2, -1])
        )

        new_s3 = tirx.Var("new_s3", "int64")
        new_s4 = tirx.Var("new_s4", "int64")

        image_features = op.wrap_nested(
            relax.BlockBuilder()
            .current()
            .match_cast(
                image_features._expr,
                relax.TensorType(
                    [
                        image_features.shape[0],
                        new_s3,
                        image_features.shape[2],
                        new_s4,
                        image_features.shape[4],
                        image_features.shape[5],
                    ],
                    image_features.dtype,
                ),
            ),
            "image_features_2",
        )

        image_features = nn.op.permute_dims(image_features, axes=([0, 1, 3, 2, 4, 5]))
        image_features_hd = nn.op.reshape(
            image_features, ([num_images, h_crop * H // 2, w_crop * H // 2, 4 * C])
        )

        return image_features_hd

    def add_image_newline(self, image_features_hd):
        """
        image_features_hd: (num_images, h_crop*12, w_crop*12, 4096)
        output: (num_images, (h_crop*12) * (w_crop*12+1), 4096)
        """
        num_images, h, w, hid_dim = image_features_hd.shape
        temp_sub_GN = self.dyn_repeat_4d_tensor(
            self.sub_GN, T.int64(1), T.int64(h), T.int64(1), T.int64(1)
        )
        image_features_hd_newline = self.dyn_concate_dim_2(image_features_hd, temp_sub_GN)
        image_features_hd_newline = nn.op.reshape(
            image_features_hd_newline, ([num_images, -1, hid_dim])
        )
        return image_features_hd_newline

    def forward(self, pixel_values: Tensor, h_crop, w_crop) -> Tensor:
        img_features = self.get_img_features(pixel_values)
        img_features = nn.op.split(img_features, indices_or_sections=[1], axis=0)

        global_image_features = img_features[0]
        global_image_features_hd = self.reshape_hd_patches_2x2merge(global_image_features, 1, 1)
        global_image_features_hd_newline = self.add_image_newline(global_image_features_hd)

        sub_image_features = img_features[1]
        sub_image_features_hd = self.reshape_hd_patches_2x2merge(sub_image_features, h_crop, w_crop)
        sub_image_features_hd_newline = self.add_image_newline(sub_image_features_hd)

        global_image_features_hd = nn.op.squeeze(global_image_features_hd_newline, 0)

        combined_image = self.dyn_concate_dim_1(sub_image_features_hd_newline, self.glb_GN)
        combined_image = self.dyn_concate_dim_1(combined_image, global_image_features_hd_newline)
        combined_image = nn.op.squeeze(combined_image, 0)

        new_s7 = tirx.Var("new_s7", "int64")

        combined_image = op.wrap_nested(
            relax.BlockBuilder()
            .current()
            .match_cast(
                combined_image._expr,
                relax.TensorType([new_s7, combined_image.shape[1]], combined_image.dtype),
            ),
            "combined_image",
        )
        output_image = self.img_projection(combined_image)
        return output_image
