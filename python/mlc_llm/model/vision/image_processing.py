"""
Implements the CLIP Image processor.
"""

from tvm import tir
from tvm.relax.frontend.nn import Module, Tensor, op
from tvm.script import tir as T


def _var(dtype):
    return T.alloc_buffer((1,), dtype, scope="local")


# pylint: disable=invalid-name,missing-docstring,no-else-return,too-many-locals,useless-parent-delegation
class ImageProcessor(Module):
    def __init__(self):
        super().__init__()

    def resize(self, image: Tensor, params):
        def get_output_image_size(image: Tensor):
            if 4 == image.ndim:
                h = image.shape[1]
                w = image.shape[2]
            elif 3 == image.ndim:
                h = image.shape[0]
                w = image.shape[1]
            else:
                assert False, "not supported image shape"

            if "height" in params and "width" in params:
                return (params["height"], params["width"])
            elif "shortest_edge" in params:
                short = tir.Select(w > h, w, h)
                long = tir.Select(w > h, h, w)
                requested_new_short = params["shortest_edge"]
                new_short, new_long = tir.generic.cast(
                    requested_new_short, "int64"
                ), tir.generic.cast(requested_new_short * tir.div(long, short), "int64")
                ret_h = tir.Select(w <= h, new_long, new_short)
                ret_w = tir.Select(w <= h, new_short, new_long)
                return (ret_h, ret_w)
            elif "hd_transform" in params:
                hd_num = 16 if "hd_num" not in params else params["hd_num"]
                pad_num = 336 if "pad_num" not in params else params["pad_num"]
                ratio = tir.Select(
                    w > h,
                    tir.div(tir.generic.cast(w, "float32"), tir.generic.cast(h, "float32")),
                    tir.div(tir.generic.cast(h, "float32"), tir.generic.cast(w, "float32")),
                )
                scale = tir.floordiv(
                    -1 + tir.generic.cast(tir.sqrt(1 + 4 * hd_num * ratio), "int64"), 2
                )
                new_w = tir.Select(
                    w >= h,
                    scale * pad_num,
                    tir.generic.cast(tir.div(scale * pad_num, ratio), "int64"),
                )
                new_h = tir.Select(
                    w >= h, tir.generic.cast(tir.div(new_w, ratio), "int64"), scale * pad_num
                )
                return (new_h, new_w)
            else:
                assert False, "not supported resize parameter"

        (new_h, new_w) = get_output_image_size(image)
        if 3 == image.ndim:
            image = op.unsqueeze(image, 0)
        out = op.interpolate(image, (new_h, new_w), data_layout="NHWC", mode="bicubic")
        return out

    # pylint: disable=too-many-arguments,too-many-locals
    def crop(self, image: Tensor, crop_size):
        def create_crop_func(dtype):
            @T.prim_func
            def crop_func(
                image: T.handle,
                out: T.handle,
                top: T.int64(),
                bottom: T.int64(),
                left: T.int64(),
                right: T.int64(),
            ):
                T.func_attr({"op_pattern": 8, "tir.noalias": True, "tir.is_scheduled": 1})
                n, c, h, w = T.int64(), T.int64(), T.int64(), T.int64()
                image_buf = T.match_buffer(image, (n, h, w, c), dtype=dtype)
                out_buf = T.match_buffer(out, (n, bottom - top, right - left, c), dtype=dtype)
                with T.block("root"):
                    for n_idx in T.thread_binding(n, thread="blockIdx.x"):
                        for h_idx in range((bottom - top)):
                            for w_idx in range((right - left)):
                                for c_idx in range(c):
                                    with T.block("compute"):
                                        T.writes(out_buf[n_idx, h_idx, w_idx, c_idx])
                                        out_buf[n_idx, h_idx, w_idx, c_idx] = image_buf[
                                            n_idx, h_idx + top, w_idx + left, c_idx
                                        ]

            return crop_func

        n, orig_height, orig_width, c = image.shape
        assert n == 1
        crop_height = crop_size["height"]
        crop_width = crop_size["width"]

        top = (orig_height - crop_height) // 2
        bottom = top + crop_height
        left = (orig_width - crop_width) // 2
        right = left + crop_width
        new_height = bottom - top
        new_width = right - left
        out = op.tensor_ir_op(
            create_crop_func(image.dtype),
            "crop",
            [image, top, bottom, left, right],
            [Tensor.placeholder([n, new_height, new_width, c], image.dtype)],
        )
        return out

    def rescale(self, image: Tensor, rescale_factor=1 / 255.0, o_dtype="float32"):
        def create_rescale_func(rescale_factor, dtype, o_dtype):
            @T.prim_func
            def rescale_func(image: T.handle, out: T.handle):
                T.func_attr({"op_pattern": 8, "tir.noalias": True, "tir.is_scheduled": 1})
                n, c, h, w = T.int64(), T.int64(), T.int64(), T.int64()
                image_buf = T.match_buffer(image, (n, h, w, c), dtype=dtype)
                out_buf = T.match_buffer(out, (n, h, w, c), dtype=o_dtype)
                for n_idx in T.thread_binding(n, thread="blockIdx.x"):
                    for h_idx, w_idx, c_idx in T.grid(h, w, c):
                        with T.block("compute"):
                            T.reads(image_buf[n_idx, h_idx, w_idx, c_idx])
                            T.writes(out_buf[n_idx, h_idx, w_idx, c_idx])
                            out_buf[n_idx, h_idx, w_idx, c_idx] = (
                                T.cast(image_buf[n_idx, h_idx, w_idx, c_idx], o_dtype)
                                * rescale_factor
                            )

            return rescale_func

        out = op.tensor_ir_op(
            create_rescale_func(rescale_factor, image.dtype, o_dtype),
            "rescale",
            [image],
            [Tensor.placeholder(image.shape, o_dtype)],
        )
        return out

    def normalize(self, image: Tensor, o_dtype="float32"):
        def create_normalize_func(dtype, o_dtype):
            @T.prim_func
            def normalize_func(image: T.handle, out: T.handle):
                T.func_attr({"op_pattern": 8, "tir.noalias": True, "tir.is_scheduled": 1})
                n, c, h, w = T.int64(), T.int64(), T.int64(), T.int64()
                image_buf = T.match_buffer(image, (n, h, w, c), dtype=dtype)
                out_buf = T.match_buffer(out, (n, h, w, c), dtype=o_dtype)
                mean = _var(o_dtype)
                stddev = _var(o_dtype)
                for n_idx in T.thread_binding(n, thread="blockIdx.x"):
                    for h_idx, w_idx, c_idx in T.grid(h, w, c):
                        with T.block("compute"):
                            T.reads(image_buf[n_idx, h_idx, w_idx, c_idx])
                            T.writes(out_buf[n_idx, h_idx, w_idx, c_idx])
                            if 0 == c_idx:
                                mean[0] = 0.48145466
                                stddev[0] = 0.26862954
                            elif 1 == c_idx:
                                mean[0] = 0.4578275
                                stddev[0] = 0.26130258
                            elif 2 == c_idx:
                                mean[0] = 0.40821073
                                stddev[0] = 0.27577711

                            out_buf[n_idx, h_idx, w_idx, c_idx] = (
                                T.cast(image_buf[n_idx, h_idx, w_idx, c_idx], o_dtype) - mean[0]
                            ) / stddev[0]

            return normalize_func

        out = op.tensor_ir_op(
            create_normalize_func(image.dtype, o_dtype),
            "normalize",
            [image],
            [Tensor.placeholder(image.shape, o_dtype)],
        )
        return out

    def pad(self, image: Tensor, dtype="uint8"):
        def create_pad_func(l, r, fill=255):
            @T.prim_func
            def pad_func(image: T.handle, out: T.handle, t: T.int64(), b: T.int64()):
                T.func_attr({"op_pattern": 8, "tir.noalias": True, "tir.is_scheduled": 1})
                n, c, h, w = T.int64(), T.int64(), T.int64(), T.int64()
                image_buf = T.match_buffer(image, (n, h, w, c), dtype=dtype)
                out_buf = T.match_buffer(out, (n, h + t + b, w + l + r, c), dtype=dtype)

                for n_idx in T.thread_binding(n, thread="blockIdx.x"):
                    for h_idx, w_idx, c_idx in T.grid(h + t + b, w + l + r, c):
                        with T.block("compute"):
                            T.reads(image_buf[n_idx, h_idx, w_idx, c_idx])
                            T.writes(out_buf[n_idx, h_idx, w_idx, c_idx])
                            if h_idx < t or h_idx > h + b or w_idx < l or w_idx > w + r:
                                out_buf[n_idx, h_idx, w_idx, c_idx] = fill
                            else:
                                out_buf[n_idx, h_idx, w_idx, c_idx] = image_buf[
                                    n_idx, h_idx - t, w_idx - l, c_idx
                                ]

            return pad_func

        h = image.shape[1]
        tar = tir.truncdiv(h + 335, 336) * 336
        t = tir.div(tar - h, 2)
        b = tar - h - t

        n, h, w, c = image.shape
        l, t, r, b = 0, t, 0, b
        out = op.tensor_ir_op(
            create_pad_func(l, r),
            "pad",
            [image, t, b],
            [Tensor.placeholder((n, tar, w, c), image.dtype)],
        )
        return out
