import tvm
from tvm import IRModule
from tvm import relax, te, tir
from tvm.relax.dpl.pattern import is_op, wildcard


@relax.expr_functor.mutator
class TransposeMatmulCodeGenerator(relax.PyExprMutator):
    def __init__(self, mod):
        super().__init__(mod)

    @staticmethod
    def pattern():
        w = wildcard()
        x = wildcard()
        wT = is_op("relax.permute_dims")(w)
        o = is_op("relax.matmul")(x, wT)
        annotations = {"o": o, "w": w, "x": x, "wT": wT}

        def _check(context: relax.transform.PatternCheckContext) -> bool:
            transpose_call = context.annotated_expr["wT"]
            ndim = transpose_call.args[0].struct_info.ndim
            if ndim == -1:
                return False
            if ndim == 2 and transpose_call.attrs.axes is None:
                return True
            axes = list(range(ndim))
            axes[-1], axes[-2] = axes[-2], axes[-1]
            return list(transpose_call.attrs.axes) == axes

        return o, annotations, _check

    def visit_call_(self, call: relax.Call) -> relax.Expr:
        def te_transposed_matmul(a: te.Tensor, b: te.Tensor) -> te.Tensor:
            a_shape = list(a.shape)
            b_shape = list(b.shape)
            a_prepended = False
            b_appended = False
            if len(a_shape) == 1:
                a_prepended = True
                a_shape.insert(0, 1)
            if len(b_shape) == 1:
                b_appended = True
                b_shape.append(1)

            is_a_larger = len(a_shape) > len(b_shape)
            offset = len(a_shape) - len(b_shape) if is_a_larger else len(b_shape) - len(a_shape)

            a_relax = relax.Var("a", relax.TensorStructInfo(a.shape))
            bT_shape = list(b.shape)
            bT_shape[-1], bT_shape[-2] = bT_shape[-2], bT_shape[-1]
            bT_relax = relax.Var("b", relax.TensorStructInfo(bT_shape))
            output_shape = self.builder_.normalize(
                relax.op.matmul(a_relax, bT_relax)
            ).struct_info.shape

            def matmul_compute(*idx_spatial):
                k = te.reduce_axis((0, a_shape[-1]), name="k")

                def multiply_compute(idx_reduce):
                    a_indices = []
                    b_indices = []

                    for i in range(offset):
                        if is_a_larger:
                            a_indices.append(idx_spatial[i])
                        else:
                            b_indices.append(idx_spatial[i])
                    for i in range(offset, len(output_shape) - (2 - a_prepended - b_appended)):
                        a_dim = a_shape[i if is_a_larger else i - offset]
                        b_dim = b_shape[i if not is_a_larger else i - offset]
                        dim_equal = a_dim == b_dim
                        if not isinstance(dim_equal, tir.IntImm) or dim_equal == 0:
                            a_dim_is_one = isinstance(a_dim, tir.IntImm) and a_dim == 1
                            b_dim_is_one = isinstance(b_dim, tir.IntImm) and b_dim == 1
                            a_indices.append(0 if a_dim_is_one else idx_spatial[i])
                            b_indices.append(0 if b_dim_is_one else idx_spatial[i])
                        else:
                            a_indices.append(idx_spatial[i])
                            b_indices.append(idx_spatial[i])

                    if not a_prepended:
                        a_indices.append(idx_spatial[-2 + b_appended])
                    a_indices.append(idx_reduce)
                    if not b_appended:
                        b_indices.append(idx_spatial[-1])
                    b_indices.append(idx_reduce)

                    dtype = a.dtype
                    if dtype != "":
                        return a(*a_indices).astype(dtype) * b(*b_indices).astype(dtype)
                    else:
                        return a(*a_indices) * b(*b_indices)

                return te.sum(multiply_compute(k), axis=k)

            return te.compute(
                output_shape,
                lambda *idx: matmul_compute(*idx),  # pylint: disable=unnecessary-lambda
                name="NT_matmul",
            )

        if isinstance(call.op, relax.GlobalVar):
            function = self.builder_.get()[call.op]
            if (
                "Composite" in function.attrs
                and function.attrs["Composite"] == "transpose_matmul_fuse"
            ):
                return self.builder_.call_te(
                    te_transposed_matmul,
                    call.args[1],
                    call.args[0],
                    primfunc_name_hint="NT_matmul",
                )

        return super().visit_call_(call)


@tvm.transform.module_pass(opt_level=0, name="FuseTransposeMatmul")
class FuseTransposeMatmul:
    def transform_module(self, mod: IRModule, ctx: tvm.transform.PassContext) -> IRModule:
        mod = relax.transform.FuseOpsByPattern(
            [("transpose_matmul_fuse", *TransposeMatmulCodeGenerator.pattern())]
        )(mod)

        transpose_matmul_codegen = TransposeMatmulCodeGenerator(mod)
        for gv in mod.functions:
            func = mod[gv]
            if not isinstance(func, relax.Function):
                continue
            func = transpose_matmul_codegen.visit_expr(func)
            transpose_matmul_codegen.builder_.update_func(gv, func)

        return transpose_matmul_codegen.builder_.get()
