"""A compiler pass that fuses dequantize matmul + epilogue."""

import operator
from functools import reduce

import tvm
from tvm import IRModule, relax
from tvm.relax.dpl import rewrite_call
from tvm.relax.dpl.pattern import is_op, wildcard


@tvm.transform.module_pass(opt_level=0, name="FuseDequantizeEpilogue")
class FuseFTDequantizeEpilogue:  # pylint: disable=too-few-public-methods
    """A compiler pass that fuses FasterTransformer dequantize matmul + epilogue."""

    def transform_module(
        self,
        mod: IRModule,
        _ctx: tvm.transform.PassContext,
    ) -> IRModule:
        """IRModule-level transformation"""
        for gv, func in mod.functions_items():
            if isinstance(func, relax.Function):
                func = fuse_bias(func)
                func = fuse_activation(func)
                func = fuse_residual_binary(func)
                func = fuse_residual_unary(func)
                mod[gv] = func
        return mod


def fuse_bias(func: relax.Function) -> relax.Function:
    """
    Fuse following `relax.add` into fastertransformer.gemm_fp16_int as bias:

    Before:
    ```
    lv1 = relax.call_dps_packed("fastertransformer.gemm_fp16_int", ...)
    lv2 = relax.add(lv1, bias)

    ```
    After:
    ```
    lv2 = relax.call_dps_packed("fastertransformer.gemm_fp16_int_bias", ..., bias, ...)
    ```

    Parameters
    ----------
    func : relax.Function
        The function before fusion.

    Returns
    -------
    ret : relax.Function
        The function after fusion.
    """
    decode_matmul = is_op("relax.call_dps_packed")(varg_default_wildcard=True)
    bias = wildcard()
    pattern = is_op("relax.add")(decode_matmul, bias) | is_op("relax.add")(bias, decode_matmul)

    def rewriter(expr, match):
        if match[decode_matmul].args[0].global_symbol == "fastertransformer.gemm_fp16_int":
            assert len(match[decode_matmul].args) == 2
            args_list = match[decode_matmul].args[1]
            assert len(args_list) == 8
            if not args_list[3].value == "identity":
                # bias cannot be fused after activation
                return expr
            matched_bias = match[bias]
            bias_stride = (
                matched_bias.struct_info.shape[-1]
                if bias
                and not reduce(operator.mul, matched_bias.struct_info.shape, 1)
                == matched_bias.struct_info.shape[-1]
                else 0
            )
            return relax.call_dps_packed(
                "fastertransformer.gemm_fp16_int_bias",
                [
                    args_list[0],  # x
                    args_list[1],  # weight
                    args_list[2],  # scale
                    matched_bias,  # bias
                    args_list[3],  # activation
                    args_list[4],  # m
                    args_list[5],  # n
                    args_list[6],  # k
                    args_list[7],  # group_size
                    bias_stride,  # bias_stride
                ],
                out_sinfo=match[decode_matmul].struct_info,
            )
        return expr

    return rewrite_call(pattern, rewriter, func)


def fuse_activation(func: relax.Function) -> relax.Function:
    """
    Fuse following `relax.nn.silu/relu/gelu` into fastertransformer.gemm_fp16_int_bias
    as activation:

    Before:
    ```
    lv1 = relax.call_dps_packed("fastertransformer.gemm_fp16_int_bias", ...)
    lv2 = relax.silu(lv1)

    ```
    After:
    ```
    lv2 = relax.call_dps_packed("fastertransformer.gemm_fp16_int_bias", ..., "silu", ...)
    ```

    Parameters
    ----------
    func : relax.Function
        The function before fusion.

    Returns
    -------
    ret : relax.Function
        The function after fusion.
    """
    # pylint: disable=unsupported-binary-operation
    decode_matmul = is_op("relax.call_dps_packed")(varg_default_wildcard=True)
    pattern = (
        is_op("relax.nn.silu")(decode_matmul)
        | is_op("relax.nn.gelu")(decode_matmul)
        | is_op("relax.nn.relu")(decode_matmul)
    )

    def rewriter(expr, match):
        if match[decode_matmul].args[0].global_symbol == "fastertransformer.gemm_fp16_int":
            matched_activation = match[pattern]
            assert matched_activation.op.name in ["relax.nn.silu", "relax.nn.gelu", "relax.nn.relu"]
            assert len(match[decode_matmul].args) == 2
            args_list = match[decode_matmul].args[1]
            assert len(args_list) == 8
            return relax.call_dps_packed(
                "fastertransformer.gemm_fp16_int",
                [
                    args_list[0],  # x
                    args_list[1],  # weight
                    args_list[2],  # scale
                    matched_activation.op.name[9:],  # activation
                    args_list[4],  # m
                    args_list[5],  # n
                    args_list[6],  # k
                    args_list[7],  # group_size
                ],
                out_sinfo=match[decode_matmul].struct_info,
            )
        if match[decode_matmul].args[0].global_symbol == "fastertransformer.gemm_fp16_int_bias":
            matched_activation = match[pattern]
            assert matched_activation.op.name in ["relax.nn.silu", "relax.nn.gelu", "relax.nn.relu"]
            assert len(match[decode_matmul].args) == 2
            args_list = match[decode_matmul].args[1]
            assert len(args_list) == 10
            return relax.call_dps_packed(
                "fastertransformer.gemm_fp16_int_bias",
                [
                    args_list[0],  # x
                    args_list[1],  # weight
                    args_list[2],  # scale
                    args_list[3],  # bias
                    matched_activation.op.name[9:],  # activation
                    args_list[5],  # m
                    args_list[6],  # n
                    args_list[7],  # k
                    args_list[8],  # group_size
                    args_list[9],  # bias_stride
                ],
                out_sinfo=match[decode_matmul].struct_info,
            )
        return expr

    return rewrite_call(pattern, rewriter, func)


def fuse_residual_binary(func: relax.Function) -> relax.Function:
    """
    Fuse following `relax.add/multiply` into fastertransformer.gemm_fp16_int_bias as
    residual binary operation:

    Before:
    ```
    lv1 = relax.call_dps_packed("fastertransformer.gemm_fp16_int_bias", ...)
    lv2 = relax.add(lv1, residual)

    ```
    After:
    ```
    lv2 = relax.call_dps_packed(
        "fastertransformer.gemm_fp16_int_bias_residual",
        ...,
        residual,
        ...,
        "plus",
        ...
    )
    ```

    Parameters
    ----------
    func : relax.Function
        The function before fusion.

    Returns
    -------
    ret : relax.Function
        The function after fusion.
    """
    # pylint: disable=unsupported-binary-operation
    decode_matmul = is_op("relax.call_dps_packed")(varg_default_wildcard=True)
    residual = wildcard()
    pattern = (
        is_op("relax.add")(decode_matmul, residual)
        | is_op("relax.add")(residual, decode_matmul)
        | is_op("relax.multiply")(decode_matmul, residual)
        | is_op("relax.multiply")(residual, decode_matmul)
    )

    def rewriter(expr, match):
        if match[decode_matmul].args[0].global_symbol == "fastertransformer.gemm_fp16_int_bias":
            matched_binary = match[pattern]
            assert matched_binary.op.name in ["relax.add", "relax.multiply"]
            binary_op = "plus" if matched_binary.op.name == "relax.add" else "multiply"
            assert len(match[decode_matmul].args) == 2
            args_list = match[decode_matmul].args[1]
            assert len(args_list) == 10
            matched_residual = match[residual]
            if not args_list[9].value == 0:
                # fastertransformer.gemm_fp16_int_bias_residual does not support
                # bias_stride != 0 yet
                return expr
            return relax.call_dps_packed(
                "fastertransformer.gemm_fp16_int_bias_residual",
                [
                    args_list[0],  # x
                    args_list[1],  # weight
                    args_list[2],  # scale
                    args_list[3],  # bias
                    matched_residual,  # residual
                    args_list[4],  # activation
                    binary_op,  # binary_op
                    "identity",  # unary_op
                    args_list[5],  # m
                    args_list[6],  # n
                    args_list[7],  # k
                    args_list[8],  # group_size
                ],
                out_sinfo=match[decode_matmul].struct_info,
            )
        return expr

    return rewrite_call(pattern, rewriter, func)


def fuse_residual_unary(func: relax.Function) -> relax.Function:
    """
    Fuse following `relax.nn.silu/relu/gelu` into fastertransformer.gemm_fp16_int_bias_residual
    as residual unary operation:

    Before:
    ```
    lv1 = relax.call_dps_packed("fastertransformer.gemm_fp16_int_bias_residual", ...)
    lv2 = relax.silu(lv1)

    ```
    After:
    ```
    lv2 = relax.call_dps_packed("fastertransformer.gemm_fp16_int_bias_residual", ..., "silu", ...)
    ```

    Parameters
    ----------
    func : relax.Function
        The function before fusion.

    Returns
    -------
    ret : relax.Function
        The function after fusion.
    """
    # pylint: disable=unsupported-binary-operation
    decode_matmul = is_op("relax.call_dps_packed")(varg_default_wildcard=True)
    pattern = (
        is_op("relax.nn.silu")(decode_matmul)
        | is_op("relax.nn.gelu")(decode_matmul)
        | is_op("relax.nn.relu")(decode_matmul)
    )

    def rewriter(expr, match):
        if (
            match[decode_matmul].args[0].global_symbol
            == "fastertransformer.gemm_fp16_int_bias_residual"
        ):
            matched_activation = match[pattern]
            assert matched_activation.op.name in ["relax.nn.silu", "relax.nn.gelu", "relax.nn.relu"]
            assert len(match[decode_matmul].args) == 2
            args_list = match[decode_matmul].args[1]
            assert len(args_list) == 12
            return relax.call_dps_packed(
                "fastertransformer.gemm_fp16_int_bias_residual",
                [
                    args_list[0],  # x
                    args_list[1],  # weight
                    args_list[2],  # scale
                    args_list[3],  # bias
                    args_list[4],  # residual
                    args_list[5],  # activation
                    args_list[6],  # binary_op
                    matched_activation.op.name[9:],  # activation
                    args_list[8],  # m
                    args_list[9],  # n
                    args_list[10],  # k
                    args_list[11],  # group_size
                ],
                out_sinfo=match[decode_matmul].struct_info,
            )
        return expr

    return rewrite_call(pattern, rewriter, func)
