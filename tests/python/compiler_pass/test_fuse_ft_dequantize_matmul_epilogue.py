# pylint: disable=invalid-name,missing-docstring,too-few-public-methods
import tvm
from tvm.ir import assert_structural_equal
from tvm.script import ir as I
from tvm.script import relax as R

from mlc_llm.compiler_pass.fuse_ft_dequantize_matmul_epilogue import (
    FuseFTDequantizeEpilogue,
)


def test_fuse_bias():
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((1, 1, 4096), "float16"),
            weight: R.Tensor((4096, 512), "int8"),
            scale: R.Tensor((1, 1024), "float16"),
            bias: R.Tensor((1, 1, 1024), "float16"),
        ):
            with R.dataflow():
                lv1 = R.call_dps_packed(
                    "fastertransformer.gemm_fp16_int",
                    (
                        x,
                        weight,
                        scale,
                        "identity",
                        R.prim_value(1),
                        R.prim_value(1024),
                        R.prim_value(4096),
                        R.prim_value(4096),
                    ),
                    out_sinfo=R.Tensor((1, 1, 1024), "float16"),
                )
                lv2 = R.add(lv1, bias)
                R.output(lv2)
            return lv2

    @I.ir_module
    class After:
        @R.function
        def main(
            x: R.Tensor((1, 1, 4096), "float16"),
            weight: R.Tensor((4096, 512), "int8"),
            scale: R.Tensor((1, 1024), "float16"),
            bias: R.Tensor((1, 1, 1024), "float16"),
        ) -> R.Tensor((1, 1, 1024), "float16"):
            with R.dataflow():
                lv2 = R.call_dps_packed(
                    "fastertransformer.gemm_fp16_int_bias",
                    (
                        x,
                        weight,
                        scale,
                        bias,
                        R.str("identity"),
                        R.prim_value(1),
                        R.prim_value(1024),
                        R.prim_value(4096),
                        R.prim_value(4096),
                        R.prim_value(0),
                    ),
                    out_sinfo=R.Tensor((1, 1, 1024), "float16"),
                )
                R.output(lv2)
            return lv2

    seq = tvm.transform.Sequential([FuseFTDequantizeEpilogue()])
    mod = seq(Before)
    assert_structural_equal(mod, After)


def test_fuse_activation():
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((1, 1, 4096), "float16"),
            weight: R.Tensor((4096, 512), "int8"),
            scale: R.Tensor((1, 1024), "float16"),
        ):
            with R.dataflow():
                lv1 = R.call_dps_packed(
                    "fastertransformer.gemm_fp16_int",
                    (
                        x,
                        weight,
                        scale,
                        "identity",
                        R.prim_value(1),
                        R.prim_value(1024),
                        R.prim_value(4096),
                        R.prim_value(4096),
                    ),
                    out_sinfo=R.Tensor((1, 1, 1024), "float16"),
                )
                lv2 = R.nn.silu(lv1)
                R.output(lv2)
            return lv2

    @I.ir_module
    class After:
        @R.function
        def main(
            x: R.Tensor((1, 1, 4096), "float16"),
            weight: R.Tensor((4096, 512), "int8"),
            scale: R.Tensor((1, 1024), "float16"),
        ) -> R.Tensor((1, 1, 1024), "float16"):
            with R.dataflow():
                lv2 = R.call_dps_packed(
                    "fastertransformer.gemm_fp16_int",
                    (
                        x,
                        weight,
                        scale,
                        R.str("silu"),
                        R.prim_value(1),
                        R.prim_value(1024),
                        R.prim_value(4096),
                        R.prim_value(4096),
                    ),
                    out_sinfo=R.Tensor((1, 1, 1024), "float16"),
                )
                R.output(lv2)
            return lv2

    seq = tvm.transform.Sequential([FuseFTDequantizeEpilogue()])
    mod = seq(Before)
    assert_structural_equal(mod, After)


def test_fuse_bias_activation():
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((1, 1, 4096), "float16"),
            weight: R.Tensor((4096, 512), "int8"),
            scale: R.Tensor((1, 1024), "float16"),
            bias: R.Tensor((1, 1, 1024), "float16"),
        ):
            with R.dataflow():
                lv1 = R.call_dps_packed(
                    "fastertransformer.gemm_fp16_int",
                    (
                        x,
                        weight,
                        scale,
                        "identity",
                        R.prim_value(1),
                        R.prim_value(1024),
                        R.prim_value(4096),
                        R.prim_value(4096),
                    ),
                    out_sinfo=R.Tensor((1, 1, 1024), "float16"),
                )
                lv2 = R.add(lv1, bias)
                lv3 = R.nn.relu(lv2)
                R.output(lv3)
            return lv3

    @I.ir_module
    class After:
        @R.function
        def main(
            x: R.Tensor((1, 1, 4096), "float16"),
            weight: R.Tensor((4096, 512), "int8"),
            scale: R.Tensor((1, 1024), "float16"),
            bias: R.Tensor((1, 1, 1024), "float16"),
        ) -> R.Tensor((1, 1, 1024), "float16"):
            with R.dataflow():
                lv2 = R.call_dps_packed(
                    "fastertransformer.gemm_fp16_int_bias",
                    (
                        x,
                        weight,
                        scale,
                        bias,
                        R.str("relu"),
                        R.prim_value(1),
                        R.prim_value(1024),
                        R.prim_value(4096),
                        R.prim_value(4096),
                        R.prim_value(0),
                    ),
                    out_sinfo=R.Tensor((1, 1, 1024), "float16"),
                )
                R.output(lv2)
            return lv2

    seq = tvm.transform.Sequential([FuseFTDequantizeEpilogue()])
    mod = seq(Before)
    assert_structural_equal(mod, After)


def test_fuse_residual_binary():
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((1, 1, 4096), "float16"),
            weight: R.Tensor((4096, 512), "int8"),
            scale: R.Tensor((1, 1024), "float16"),
            bias: R.Tensor((1, 1, 1024), "float16"),
            residual: R.Tensor((1, 1, 1024), "float16"),
        ):
            with R.dataflow():
                lv1 = R.call_dps_packed(
                    "fastertransformer.gemm_fp16_int",
                    (
                        x,
                        weight,
                        scale,
                        "identity",
                        R.prim_value(1),
                        R.prim_value(1024),
                        R.prim_value(4096),
                        R.prim_value(4096),
                    ),
                    out_sinfo=R.Tensor((1, 1, 1024), "float16"),
                )
                lv2 = R.add(lv1, bias)
                lv3 = R.nn.relu(lv2)
                lv4 = R.multiply(lv3, residual)
                R.output(lv4)
            return lv4

    @I.ir_module
    class After:
        @R.function
        def main(
            x: R.Tensor((1, 1, 4096), "float16"),
            weight: R.Tensor((4096, 512), "int8"),
            scale: R.Tensor((1, 1024), "float16"),
            bias: R.Tensor((1, 1, 1024), "float16"),
            residual: R.Tensor((1, 1, 1024), "float16"),
        ) -> R.Tensor((1, 1, 1024), "float16"):
            with R.dataflow():
                lv2 = R.call_dps_packed(
                    "fastertransformer.gemm_fp16_int_bias_residual",
                    (
                        x,
                        weight,
                        scale,
                        bias,
                        residual,
                        R.str("relu"),
                        R.str("multiply"),
                        R.str("identity"),
                        R.prim_value(1),
                        R.prim_value(1024),
                        R.prim_value(4096),
                        R.prim_value(4096),
                    ),
                    out_sinfo=R.Tensor((1, 1, 1024), "float16"),
                )
                R.output(lv2)
            return lv2

    seq = tvm.transform.Sequential([FuseFTDequantizeEpilogue()])
    mod = seq(Before)
    assert_structural_equal(mod, After)


def test_fuse_residual_unary():
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((1, 1, 4096), "float16"),
            weight: R.Tensor((4096, 512), "int8"),
            scale: R.Tensor((1, 1024), "float16"),
            bias: R.Tensor((1, 1, 1024), "float16"),
            residual: R.Tensor((1, 1, 1024), "float16"),
        ):
            with R.dataflow():
                lv1 = R.call_dps_packed(
                    "fastertransformer.gemm_fp16_int",
                    (
                        x,
                        weight,
                        scale,
                        "identity",
                        R.prim_value(1),
                        R.prim_value(1024),
                        R.prim_value(4096),
                        R.prim_value(4096),
                    ),
                    out_sinfo=R.Tensor((1, 1, 1024), "float16"),
                )
                lv2 = R.add(lv1, bias)
                lv3 = R.nn.relu(lv2)
                lv4 = R.add(lv3, residual)
                lv5 = R.nn.gelu(lv4)
                R.output(lv5)
            return lv5

    @I.ir_module
    class After:
        @R.function
        def main(
            x: R.Tensor((1, 1, 4096), "float16"),
            weight: R.Tensor((4096, 512), "int8"),
            scale: R.Tensor((1, 1024), "float16"),
            bias: R.Tensor((1, 1, 1024), "float16"),
            residual: R.Tensor((1, 1, 1024), "float16"),
        ) -> R.Tensor((1, 1, 1024), "float16"):
            with R.dataflow():
                lv2 = R.call_dps_packed(
                    "fastertransformer.gemm_fp16_int_bias_residual",
                    (
                        x,
                        weight,
                        scale,
                        bias,
                        residual,
                        R.str("relu"),
                        R.str("plus"),
                        R.str("gelu"),
                        R.prim_value(1),
                        R.prim_value(1024),
                        R.prim_value(4096),
                        R.prim_value(4096),
                    ),
                    out_sinfo=R.Tensor((1, 1, 1024), "float16"),
                )
                R.output(lv2)
            return lv2

    seq = tvm.transform.Sequential([FuseFTDequantizeEpilogue()])
    mod = seq(Before)
    assert_structural_equal(mod, After)


if __name__ == "__main__":
    test_fuse_bias()
    test_fuse_activation()
    test_fuse_bias_activation()
    test_fuse_residual_binary()
    test_fuse_residual_unary()
