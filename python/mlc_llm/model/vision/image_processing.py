"""
Implements the CLIP Image processor.
"""

from tvm import tir
from tvm.relax.frontend.nn import Module, Tensor, op
from tvm.script import tir as T


def _var(dtype, size=1):
    return T.alloc_buffer((size,), dtype, scope="local")


# pylint: disable=invalid-name,missing-docstring,no-else-return,too-many-locals,useless-parent-delegation
class ImageProcessor(Module):
    def __init__(self):
        super().__init__()

    # pylint: disable=dangerous-default-value
    def apply_schedule(self, sch, block, bdx=32, tile=[32, 32]):
        loop_x, loop_y = sch.get_loops(block)[-2:]
        xo, xi = sch.split(loop_x, factors=[tile[0], None])
        yo, yi = sch.split(loop_y, factors=[tile[1], None])
        sch.reorder(xo, yo, xi, yi)
        t = sch.fuse(xo, yo)
        ty, tx = sch.split(t, factors=[None, bdx])
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")

    def resize(self, image: Tensor, params):  # image layout:NCHW
        assert 4 == image.ndim, "image should be 4D data tensor"
        assert 3 == image.shape[1], "image layout should be NCHW"

        def get_output_image_size(image: Tensor):
            h = image.shape[2]
            w = image.shape[3]

            if "height" in params and "width" in params:
                return (params["height"], params["width"])
            elif "shortest_edge" in params:
                short = tir.Select(w < h, w, h)
                long = tir.Select(w > h, w, h)
                requested_new_short = params["shortest_edge"]
                new_short, new_long = tir.generic.cast(
                    requested_new_short, "int64"
                ), tir.generic.cast(
                    requested_new_short
                    * tir.div(
                        tir.generic.cast(long, "float32"), tir.generic.cast(short, "float32")
                    ),
                    "int64",
                )
                ret_h = tir.Select(w <= h, new_long, new_short)
                ret_w = tir.Select(w <= h, new_short, new_long)
                return (ret_h, ret_w)
            elif "hd_transform" in params:
                hd_num = 4 if "hd_num" not in params else params["hd_num"]
                pad_num = 336 if "pad_num" not in params else params["pad_num"]
                ratio = tir.Select(
                    w > h,
                    tir.div(tir.generic.cast(w, "float32"), tir.generic.cast(h, "float32")),
                    tir.div(tir.generic.cast(h, "float32"), tir.generic.cast(w, "float32")),
                )

                scale = tir.ceil(tir.sqrt(tir.generic.cast(hd_num, "float32") * ratio))

                scale = tir.Select(
                    (scale * tir.ceil(tir.div(scale, ratio))) > hd_num,
                    scale - 1,
                    scale,
                )
                scale = tir.generic.cast(scale, "int64")

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
        out = op.interpolate(image, (new_h, new_w), data_layout="NCHW", mode="linear")
        return out

    # pylint: disable=too-many-arguments,too-many-locals
    def crop(self, image: Tensor, crop_size):
        assert 4 == image.ndim, "image should be 4D data tensor"
        assert 3 == image.shape[1], "image layout should be NCHW"

        def create_crop_func(dtype):  # , top, bottom, left, right):
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
                image_buf = T.match_buffer(image, (n, c, h, w), dtype=dtype)
                out_buf = T.match_buffer(out, (n, c, bottom - top, right - left), dtype=dtype)
                out_h = bottom - top
                out_w = right - left
                for n_idx in T.thread_binding(n, thread="blockIdx.x"):
                    for c_idx in T.thread_binding(c, thread="blockIdx.y"):
                        for h_idx, w_idx in T.grid(out_h, out_w):
                            with T.block("crop"):
                                if (h_idx + T.int64(top)) < h and (w_idx + T.int64(left)) < w:
                                    T.writes(out_buf[n_idx, c_idx, h_idx, w_idx])
                                    T.reads(image_buf[n_idx, c_idx, h_idx + top, w_idx + left])
                                    out_buf[n_idx, c_idx, h_idx, w_idx] = image_buf[
                                        n_idx, c_idx, h_idx + top, w_idx + left
                                    ]

            sch = tir.Schedule(crop_func)
            self.apply_schedule(sch, sch.get_block("crop"))
            return sch.mod["main"].with_attr("tir.is_scheduled", 1)

        n, c, orig_height, orig_width = image.shape
        crop_height = crop_size["height"]
        crop_width = crop_size["width"]

        top = (orig_height - crop_height) // 2
        bottom = orig_height - top

        left = (orig_width - crop_width) // 2
        right = orig_width - left

        out = op.tensor_ir_op(
            create_crop_func(image.dtype),
            "crop",
            [image, top, bottom, left, right],
            [Tensor.placeholder([n, c, crop_height, crop_width], image.dtype)],
        )
        return out

    def rescale(self, image: Tensor, rescale_factor=1 / 255.0, o_dtype="float32"):
        assert 4 == image.ndim, "image should be 4D data tensor"
        assert 3 == image.shape[1], "image layout should be NCHW"

        def create_rescale_func(rescale_factor, dtype, o_dtype):
            @T.prim_func
            def rescale_func(image: T.handle, out: T.handle):
                T.func_attr({"op_pattern": 8, "tir.noalias": True, "tir.is_scheduled": 1})
                n, c, h, w = T.int64(), T.int64(), T.int64(), T.int64()
                image_buf = T.match_buffer(image, (n, c, h, w), dtype=dtype)
                out_buf = T.match_buffer(out, (n, c, h, w), dtype=o_dtype)

                for n_idx in T.thread_binding(n, thread="blockIdx.x"):
                    for c_idx in T.thread_binding(c, thread="blockIdx.y"):
                        for h_idx, w_idx in T.grid(h, w):
                            with T.block("rescale"):
                                T.reads(image_buf[n_idx, c_idx, h_idx, w_idx])
                                T.writes(out_buf[n_idx, c_idx, h_idx, w_idx])
                                if h_idx < h and w_idx < w:
                                    out_buf[n_idx, c_idx, h_idx, w_idx] = (
                                        T.cast(image_buf[n_idx, c_idx, h_idx, w_idx], o_dtype)
                                        * rescale_factor
                                    )

            sch = tir.Schedule(rescale_func)
            self.apply_schedule(sch, sch.get_block("rescale"))
            return sch.mod["main"].with_attr("tir.is_scheduled", 1)

        out = op.tensor_ir_op(
            create_rescale_func(rescale_factor, image.dtype, o_dtype),
            "rescale",
            [image],
            [Tensor.placeholder(image.shape, o_dtype)],
        )
        return out

    def normalize(self, image: Tensor, o_dtype="float32"):
        assert 4 == image.ndim, "image should be 4D data tensor"
        assert 3 == image.shape[1], "image layout should be NCHW"

        def create_normalize_func(dtype, o_dtype):
            @T.prim_func
            def normalize_func(image: T.handle, out: T.handle):
                n, c, h, w = T.int64(), T.int64(), T.int64(), T.int64()
                image_buf = T.match_buffer(image, (n, c, h, w), dtype=dtype)
                out_buf = T.match_buffer(out, (n, c, h, w), dtype=o_dtype)
                mean = _var(o_dtype, 3)
                stddev = _var(o_dtype, 3)

                for n_idx in T.thread_binding(n, thread="blockIdx.x"):
                    for c_idx in T.thread_binding(c, thread="blockIdx.y"):
                        for h_idx, w_idx in T.grid(h, w):
                            with T.block("normalize"):
                                T.reads(
                                    image_buf[n_idx, c_idx, h_idx, w_idx],
                                    mean[c_idx],
                                    stddev[c_idx],
                                )
                                T.writes(out_buf[n_idx, c_idx, h_idx, w_idx])
                                with T.init():
                                    mean[0] = 0.48145466
                                    stddev[0] = 0.26862954
                                    mean[1] = 0.4578275
                                    stddev[1] = 0.26130258
                                    mean[2] = 0.40821073
                                    stddev[2] = 0.27577711
                                if h_idx < h and w_idx < w:
                                    out_buf[n_idx, c_idx, h_idx, w_idx] = (
                                        T.cast(image_buf[n_idx, c_idx, h_idx, w_idx], o_dtype)
                                        - mean[c_idx]
                                    ) / stddev[c_idx]

            sch = tir.Schedule(normalize_func)
            self.apply_schedule(sch, sch.get_block("normalize"))
            return sch.mod["main"].with_attr("tir.is_scheduled", 1)

        out = op.tensor_ir_op(
            create_normalize_func(image.dtype, o_dtype),
            "normalize",
            [image],
            [Tensor.placeholder(image.shape, o_dtype)],
        )
        return out

    def pad(self, image: Tensor, dtype="uint8"):
        assert 4 == image.ndim, "image should be 4D data tensor"
        assert 3 == image.shape[1], "image layout should be NCHW"

        def create_pad_func(l, r, fill=255):
            @T.prim_func
            def pad_func(image: T.handle, out: T.handle, t: T.int64(), b: T.int64()):
                T.func_attr({"op_pattern": 8, "tir.noalias": True, "tir.is_scheduled": 1})
                n, c, h, w = T.int64(), T.int64(), T.int64(), T.int64()
                image_buf = T.match_buffer(image, (n, c, h, w), dtype=dtype)
                out_buf = T.match_buffer(out, (n, c, h + t + b, w + l + r), dtype=dtype)
                out_h = h + t + b
                out_w = w + l + r

                for n_idx in T.thread_binding(n, thread="blockIdx.x"):
                    for c_idx in T.thread_binding(c, thread="blockIdx.y"):
                        for h_idx, w_idx in T.grid(out_h, out_w):
                            with T.block("pad"):
                                T.reads(image_buf[n_idx, c_idx, h_idx, w_idx])
                                T.writes(out_buf[n_idx, c_idx, h_idx, w_idx])
                                if h_idx < t or h_idx > h + b or w_idx < l or w_idx > w + r:
                                    out_buf[n_idx, c_idx, h_idx, w_idx] = fill
                                else:
                                    out_buf[n_idx, c_idx, h_idx, w_idx] = image_buf[
                                        n_idx, c_idx, h_idx - t, w_idx - l
                                    ]

            sch = tir.Schedule(pad_func)
            self.apply_schedule(sch, sch.get_block("pad"))
            return sch.mod["main"].with_attr("tir.is_scheduled", 1)

        h = image.shape[2]
        tar = tir.truncdiv(h + 335, 336) * 336
        t = tir.div(tar - h, 2)
        b = tar - h - t
        l = 0
        r = 0

        n, c, h, w = image.shape
        out = op.tensor_ir_op(
            create_pad_func(l, r),
            "pad",
            [image, t, b],
            [Tensor.placeholder((n, c, tar, w), image.dtype)],
        )
        return out

    def preprocess(self, pixel_values):
        return pixel_values
