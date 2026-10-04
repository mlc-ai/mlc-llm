"""
Implements the CLIP Image processor.
"""

from tvm import s_tir, tirx
from tvm.relax.frontend.nn import Module, Tensor, op
from tvm.script import s_tir as Ts
from tvm.script import tirx as T


def _var(dtype, size=1):
    return Ts.sblock_alloc_buffer((size,), dtype, scope="local")


class ImageProcessor(Module):
    def __init__(self):
        super().__init__()

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
                short = tirx.Select(w < h, w, h)
                long = tirx.Select(w > h, w, h)
                requested_new_short = params["shortest_edge"]
                new_short, new_long = (
                    tirx.Cast("int64", requested_new_short),
                    tirx.Cast(
                        "int64",
                        requested_new_short
                        * tirx.div(
                            tirx.Cast("float32", long),
                            tirx.Cast("float32", short),
                        ),
                    ),
                )
                ret_h = tirx.Select(w <= h, new_long, new_short)
                ret_w = tirx.Select(w <= h, new_short, new_long)
                return (ret_h, ret_w)
            elif "hd_transform" in params:
                hd_num = 4 if "hd_num" not in params else params["hd_num"]
                pad_num = 336 if "pad_num" not in params else params["pad_num"]
                ratio = tirx.Select(
                    w > h,
                    tirx.div(tirx.Cast("float32", w), tirx.Cast("float32", h)),
                    tirx.div(tirx.Cast("float32", h), tirx.Cast("float32", w)),
                )

                scale = tirx.ceil(tirx.sqrt(tirx.Cast("float32", hd_num) * ratio))

                scale = tirx.Select(
                    (scale * tirx.ceil(tirx.div(scale, ratio))) > hd_num,
                    scale - 1,
                    scale,
                )
                scale = tirx.Cast("int64", scale)

                new_w = tirx.Select(
                    w >= h,
                    scale * pad_num,
                    tirx.Cast("int64", tirx.div(scale * pad_num, ratio)),
                )
                new_h = tirx.Select(
                    w >= h,
                    tirx.Cast("int64", tirx.div(new_w, ratio)),
                    scale * pad_num,
                )
                return (new_h, new_w)
            else:
                assert False, "not supported resize parameter"

        new_h, new_w = get_output_image_size(image)
        out = op.interpolate(image, (new_h, new_w), data_layout="NCHW", mode="linear")
        return out

    def crop(self, image: Tensor, crop_size):
        assert 4 == image.ndim, "image should be 4D data tensor"
        assert 3 == image.shape[1], "image layout should be NCHW"

        def create_crop_func(dtype):  # , top, bottom, left, right):
            n = T.dynamic("n", "int64")
            c = T.dynamic("c", "int64")
            h = T.dynamic("h", "int64")
            w = T.dynamic("w", "int64")
            top = T.dynamic("top", "int64")
            bottom = T.dynamic("bottom", "int64")
            left = T.dynamic("left", "int64")
            right = T.dynamic("right", "int64")

            @Ts.prim_func
            def crop_func(
                image_buf: T.Tensor((n, c, h, w), dtype),
                top: T.int64(),
                bottom: T.int64(),
                left: T.int64(),
                right: T.int64(),
                out_buf: T.Tensor((n, c, bottom - top, right - left), dtype),
            ):
                T.func_attr({"op_pattern": 8, "tirx.noalias": True, "tirx.is_scheduled": 1})
                out_h = bottom - top
                out_w = right - left
                for n_idx in T.thread_binding(n, thread="blockIdx.x"):
                    for c_idx in T.thread_binding(c, thread="blockIdx.y"):
                        for h_idx, w_idx in T.grid(out_h, out_w):
                            with Ts.sblock("crop"):
                                Ts.writes(out_buf[n_idx, c_idx, h_idx, w_idx])
                                Ts.reads(image_buf[n_idx, c_idx, h_idx + top, w_idx + left])
                                if (h_idx + T.int64(top)) < h and (w_idx + T.int64(left)) < w:
                                    out_buf[n_idx, c_idx, h_idx, w_idx] = image_buf[
                                        n_idx, c_idx, h_idx + top, w_idx + left
                                    ]

            sch = s_tir.Schedule(crop_func)
            self.apply_schedule(sch, sch.get_sblock("crop"))
            return sch.mod["main"].with_attr("tirx.is_scheduled", 1)

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
            n = T.dynamic("n", "int64")
            c = T.dynamic("c", "int64")
            h = T.dynamic("h", "int64")
            w = T.dynamic("w", "int64")

            @Ts.prim_func
            def rescale_func(
                image_buf: T.Tensor((n, c, h, w), dtype),
                out_buf: T.Tensor((n, c, h, w), o_dtype),
            ):
                T.func_attr({"op_pattern": 8, "tirx.noalias": True, "tirx.is_scheduled": 1})

                for n_idx in T.thread_binding(n, thread="blockIdx.x"):
                    for c_idx in T.thread_binding(c, thread="blockIdx.y"):
                        for h_idx, w_idx in T.grid(h, w):
                            with Ts.sblock("rescale"):
                                Ts.reads(image_buf[n_idx, c_idx, h_idx, w_idx])
                                Ts.writes(out_buf[n_idx, c_idx, h_idx, w_idx])
                                if h_idx < h and w_idx < w:
                                    out_buf[n_idx, c_idx, h_idx, w_idx] = (
                                        T.cast(
                                            image_buf[n_idx, c_idx, h_idx, w_idx],
                                            o_dtype,
                                        )
                                        * rescale_factor
                                    )

            sch = s_tir.Schedule(rescale_func)
            self.apply_schedule(sch, sch.get_sblock("rescale"))
            return sch.mod["main"].with_attr("tirx.is_scheduled", 1)

        out = op.tensor_ir_op(
            create_rescale_func(rescale_factor, image.dtype, o_dtype),
            "rescale",
            [image],
            [Tensor.placeholder(image.shape, o_dtype)],
        )
        return out

    def _normalize_impl(self, image: Tensor, mean, std, name, o_dtype="float32"):
        assert 4 == image.ndim, "image should be 4D data tensor"
        assert 3 == image.shape[1], "image layout should be NCHW"

        def create_normalize_func(mean_vals, std_vals, dtype, o_dtype):
            n = T.dynamic("n", "int64")
            c = T.dynamic("c", "int64")
            h = T.dynamic("h", "int64")
            w = T.dynamic("w", "int64")

            @Ts.prim_func
            def normalize_func(
                image_buf: T.Tensor((n, c, h, w), dtype),
                out_buf: T.Tensor((n, c, h, w), o_dtype),
            ):
                mean = _var(o_dtype, 3)
                stddev = _var(o_dtype, 3)

                for n_idx in T.thread_binding(n, thread="blockIdx.x"):
                    for c_idx in T.thread_binding(c, thread="blockIdx.y"):
                        for h_idx, w_idx in T.grid(h, w):
                            with Ts.sblock("normalize"):
                                Ts.reads(
                                    image_buf[n_idx, c_idx, h_idx, w_idx],
                                    mean[c_idx],
                                    stddev[c_idx],
                                )
                                Ts.writes(out_buf[n_idx, c_idx, h_idx, w_idx])
                                with Ts.init():
                                    mean[0] = mean_vals[0]
                                    stddev[0] = std_vals[0]
                                    mean[1] = mean_vals[1]
                                    stddev[1] = std_vals[1]
                                    mean[2] = mean_vals[2]
                                    stddev[2] = std_vals[2]
                                if h_idx < h and w_idx < w:
                                    out_buf[n_idx, c_idx, h_idx, w_idx] = (
                                        T.cast(
                                            image_buf[n_idx, c_idx, h_idx, w_idx],
                                            o_dtype,
                                        )
                                        - mean[c_idx]
                                    ) / stddev[c_idx]

            sch = s_tir.Schedule(normalize_func)
            self.apply_schedule(sch, sch.get_sblock("normalize"))
            return sch.mod["main"].with_attr("tirx.is_scheduled", 1)

        out = op.tensor_ir_op(
            create_normalize_func(mean, std, image.dtype, o_dtype),
            name,
            [image],
            [Tensor.placeholder(image.shape, o_dtype)],
        )
        return out

    def normalize(self, image: Tensor, o_dtype="float32"):
        return self._normalize_impl(
            image,
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
            name="normalize",
            o_dtype=o_dtype,
        )

    def normalize_siglip(self, image: Tensor, o_dtype="float32"):
        """Normalize with SigLIP values: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]"""
        return self._normalize_impl(
            image,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            name="normalize_siglip",
            o_dtype=o_dtype,
        )

    def pad(self, image: Tensor, dtype="uint8"):
        assert 4 == image.ndim, "image should be 4D data tensor"
        assert 3 == image.shape[1], "image layout should be NCHW"

        def create_pad_func(left, right, fill=255):
            n = T.dynamic("n", "int64")
            c = T.dynamic("c", "int64")
            h = T.dynamic("h", "int64")
            w = T.dynamic("w", "int64")
            t = T.dynamic("t", "int64")
            b = T.dynamic("b", "int64")

            @Ts.prim_func
            def pad_func(
                image_buf: T.Tensor((n, c, h, w), dtype),
                t: T.int64(),
                b: T.int64(),
                out_buf: T.Tensor((n, c, h + t + b, w + left + right), dtype),
            ):
                T.func_attr({"op_pattern": 8, "tirx.noalias": True, "tirx.is_scheduled": 1})
                out_h = h + t + b
                out_w = w + left + right

                for n_idx in T.thread_binding(n, thread="blockIdx.x"):
                    for c_idx in T.thread_binding(c, thread="blockIdx.y"):
                        for h_idx, w_idx in T.grid(out_h, out_w):
                            with Ts.sblock("pad"):
                                Ts.reads(image_buf[n_idx, c_idx, h_idx, w_idx])
                                Ts.writes(out_buf[n_idx, c_idx, h_idx, w_idx])
                                if h_idx < t or h_idx > h + b or w_idx < left or w_idx > w + right:
                                    out_buf[n_idx, c_idx, h_idx, w_idx] = fill
                                else:
                                    out_buf[n_idx, c_idx, h_idx, w_idx] = image_buf[
                                        n_idx, c_idx, h_idx - t, w_idx - left
                                    ]

            sch = s_tir.Schedule(pad_func)
            self.apply_schedule(sch, sch.get_sblock("pad"))
            return sch.mod["main"].with_attr("tirx.is_scheduled", 1)

        h = image.shape[2]
        tar = tirx.truncdiv(h + 335, 336) * 336
        t = tirx.div(tar - h, 2)
        b = tar - h - t
        left = 0
        right = 0

        n, c, h, w = image.shape
        out = op.tensor_ir_op(
            create_pad_func(left, right),
            "pad",
            [image, t, b],
            [Tensor.placeholder((n, c, tar, w), image.dtype)],
        )
        return out

    def preprocess(self, pixel_values):
        return pixel_values
