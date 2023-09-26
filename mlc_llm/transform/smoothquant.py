import numpy as np
import tvm
from tvm import relax
from tvm.relax.analysis import remove_all_unused
from tvm.relax.dpl import rewrite_call, is_op, wildcard
from tvm.relax.expr_functor import PyExprMutator, mutator
from tvm.script import relax as R

from typing import Dict


QSCHEMES = ("smq_q8i8f16_0", "smq_q8i8f16_1")
OPMODES = ("smoothing", *QSCHEMES)

@mutator
class Annotator(PyExprMutator):
    def __init__(self, irmod: tvm.IRModule, op_mode: str) -> None:
        super().__init__(irmod)
        self.mod = irmod
        self.sm_counter = 0
        self.new_params = []
        # Operation mode of the annotator: "smoothing" or "quantization"
        assert op_mode in OPMODES, f"unsupported operation mode: '{op_mode}'"
        self.op_mode = op_mode

    def transform(self) -> tvm.IRModule:
        for gv, func in self.mod.functions.items():
            if not isinstance(func, relax.Function):
                continue
            self.sm_counter = 0
            self.new_params = []
            updated_func = self.visit_expr(func)
            updated_func = remove_all_unused(updated_func)
            self.builder_.update_func(gv, updated_func)

        return self.builder_.get()
    
    def visit_function_(self, f):
        body = super().visit_expr(f.body)
        params = list(f.params) + list(self.new_params)
        return tvm.relax.Function(params, body, f.ret_struct_info, f.is_pure, f.attrs, f.span)

    def visit_call_(self, call: relax.Call) -> relax.Expr:
        call = super().visit_call_(call)
        if call.op != tvm.ir.Op.get("relax.matmul"):
            return call
        permute = self.lookup_binding(call.args[1])
        if permute is None or permute.op != tvm.ir.Op.get("relax.permute_dims"):
            return call
        act = call.args[0]
        weights = permute.args[0]
        if weights.struct_info.ndim != 2:
            return call
        if act.struct_info.ndim != 2 and act.struct_info.ndim != 3:
            return call

        def make_scale_param(shape: relax.ShapeExpr, dtype: str, kind: int) -> tvm.relax.Var:
            """
            Create scale parameter.

            In case of quantization:
              scale is a Tensor with the single element for per-tensor quantization scheme
              (smq_q8i8f16_0) or 1D Tensor for weights in per-channel quantization scheme.

            In case of smoothing:
              scale is a 1D Tensor with the size == dimension of reduction axis in matmul op
              (shape[-1]).
            """
            if self.op_mode == "smq_q8i8f16_0":
                n = 1 # per-tensor quantization for activations and weights
            elif self.op_mode == "smq_q8i8f16_1":
                n = 1 if kind == 1 else shape[-2]  # per-channel quantization for weights
            else:
                n = shape[-1]  # smoothing scale
            scale = relax.Var(f"sq_scale_{self.sm_counter}", relax.TensorStructInfo([n], dtype))
            self.sm_counter += 1
            self.new_params.append(scale)
            return scale

        a_scale = make_scale_param(act.struct_info.shape, act.struct_info.dtype, kind=1)
        w_scale = make_scale_param(weights.struct_info.shape, weights.struct_info.dtype, kind=2)
        if self.op_mode.startswith("smq_q8i8f16"):
            lhs = R.smooth(act, a_scale, kind=1, mode="identity")
            rhs = R.smooth(weights, w_scale, kind=2, mode="identity")
        else:
            lhs = R.divide(act, a_scale)
            rhs = R.multiply(weights, w_scale)
        return R.linear(lhs, rhs)


@tvm.transform.module_pass(opt_level=0, name="SmoothQuantAnnotator")
class SmoothQuantAnnotator:
    """
    Insert R.multiply and R.divide (or R.smooth in case of op_mode == "smq_q8i8f16_*") ops before
    R.linear. Add scales (second argument of R.multiply or R.smooth) to the list of relax.Function
    parameters.
    Example:
      R.linear(lhs, rhs)  -->  op1 = R.smooth(lhs, scale1, kind=1, mode="identity")
                               op2 = R.smooth(rhs, scale2, kind=2, mode="identity")
                               R.linear(op1, op2)

      for the self.op_mode == "smoothing" case:
      R.linear(lhs, rhs)  -->  op1 = R.divide(lhs, scale1)
                               op2 = R.multiply(rhs, scale2)
                               R.linear(op1, op2)
    """
    def __init__(self, op_mode: str = "smoothing") -> None:
        self.op_mode = op_mode
        pass

    def transform_module(self, irmod: tvm.IRModule, ctx: tvm.transform.PassContext) -> tvm.IRModule:
        return Annotator(irmod, self.op_mode).transform()


@tvm.transform.module_pass(opt_level=0, name="SmoothQuantStatCollector")
class SmoothQuantStatCollector:
    """
    This pass modifies IRModule to enable statistics collection. It does several modifications:
    1) Insert chain of simple ops (abs, max, squeeze) just after annotate operation
       (R.annotate.smooth/R.divide/R.multiply). This is done for memory footprint optimization only.
       Since we do not want to dump the whole tensor and dump already preprocessed information
       (abs -> max -> squeeze).
    2) Substitute scale param in R.annotate.smooth (or second arguemnt in R.divide/R.multiply) with
       dummy ones and remove these params from relax.Function. Dummy param is np.ones array in this
       case.
    3) Add new outputs in relax.Function that correspond to the last op from 1).
    """
    def __init__(self, op_mode: str = "smoothing") -> None:
        # Operation mode of the collector: "smoothing" or "quantization"
        assert op_mode in OPMODES, f"unsupported operation mode: '{op_mode}'"
        self.op_mode = op_mode

    def transform_module(self, mod: tvm.IRModule, ctx: tvm.transform.PassContext) -> tvm.IRModule:
        @mutator
        class ParamsAndOutputsMutator(PyExprMutator):
            def __init__(self, mod: tvm.IRModule, op_mode: str) -> None:
                super().__init__(mod)
                self.mod = mod
                # Operation mode of the annotator: "smoothing" or "quantization"
                assert op_mode in OPMODES, f"unsupported operation mode: '{op_mode}'"
                self.op_mode = op_mode
                self.var2val: Dict[relax.Var, relax.Expr] = {}
                self.profile_points = []
                self.params_to_remove = []

                attrs = {"mode": "identity"}
                self.lhs_sm = (
                    is_op("relax.annotate.smooth")(wildcard(), wildcard()).has_attr(attrs) |
                    is_op("relax.divide")(wildcard(), wildcard())
                )
                self.rhs_sm = (
                    is_op("relax.annotate.smooth")(wildcard(), wildcard()).has_attr(attrs) |
                    is_op("relax.multiply")(wildcard(), wildcard())
                )
                self.permute = is_op("relax.permute_dims")(self.rhs_sm)
                self.pattern = is_op("relax.matmul")(self.lhs_sm, self.permute)
            
            def transform(self) -> tvm.IRModule:
                for gv, func in self.mod.functions.items():
                    if not isinstance(func, relax.Function):
                        continue
                    self.var2val = tvm.relax.analysis.get_var2val(func)
                    self.profile_points = []
                    self.params_to_remove = []
                    updated_func = self.visit_expr(func)
                    updated_func = remove_all_unused(updated_func)
                    self.builder_.update_func(gv, updated_func)
                return self.builder_.get()
            
            def visit_function_(self, f):
                body = super().visit_expr(f.body)
                new_params = [param for param in f.params if param not in self.params_to_remove]
                return relax.Function(new_params, body, None, f.is_pure, f.attrs, f.span)
            
            def visit_seq_expr_(self, op: relax.SeqExpr) -> relax.Expr:
                op = super().visit_seq_expr_(op)
                if len(self.profile_points) != 0:
                    new_body = relax.Tuple([op.body, relax.Tuple(self.profile_points)])
                    return relax.SeqExpr(op.blocks, new_body, op.span)
                return op

            def visit_dataflow_block_(self, block: relax.DataflowBlock) -> relax.DataflowBlock:
                self.builder_._begin_dataflow_block()
                for binding in block.bindings:
                    self.visit_binding(binding)
                if len(self.profile_points) != 0:
                    self.builder_.emit_output(self.profile_points)
                return self.builder_._end_block()

            def visit_call_(self, call: relax.Call) -> relax.Expr:
                call = super().visit_call_(call)
                matchings = self.pattern.extract_matched_expr(call, self.var2val)
                if matchings:
                    m_smq1 = matchings[self.lhs_sm]
                    a_smq = self._emit_annotate_op(m_smq1, kind=1)
                    a_out = self._emit_abs_max_ops_chain(a_smq, kind=1)
                    m_smq2 = matchings[self.rhs_sm]
                    w_smq = self._emit_annotate_op(m_smq2, kind=2)
                    w_out = self._emit_abs_max_ops_chain(w_smq, kind=2)
                    self.profile_points.extend([a_out, w_out])
                    self.params_to_remove.extend([m_smq1.args[1], m_smq2.args[1]])
                    return self.builder_.emit(R.linear(a_smq, w_smq))

                return call

            def _emit_annotate_op(self, call: relax.Call, kind: int) -> relax.Var:
                tinfo = call.args[1].struct_info
                data = np.ones([int(dim) for dim in tinfo.shape], tinfo.dtype)
                scale = tvm.runtime.ndarray.array(data, device=tvm.cpu())
                if call.op == tvm.ir.Op.get("relax.multiply"):
                    smq = self.builder_.emit(R.multiply(call.args[0], relax.Constant(scale)))
                elif call.op == tvm.ir.Op.get("relax.divide"):
                    smq = self.builder_.emit(R.divide(call.args[0], relax.Constant(scale)))
                else:
                    smq = self.builder_.emit(
                        R.smooth(call.args[0], relax.Constant(scale), kind=kind, mode="identity")
                    )
                return smq

            def _emit_abs_max_ops_chain(self, expr: relax.Var, kind: str) -> relax.Var:
                assert expr.struct_info.ndim >= 2, "Tensor dim num should be >= 2"
                abs_expr = self.builder_.emit(R.abs(expr))
                if self.op_mode == "smq_q8i8f16_1":
                    if kind == 1:
                        max_expr = self.builder_.emit(R.max(abs_expr, axis=-2))
                    else:
                        max_expr = self.builder_.emit(R.max(abs_expr, axis=-1))
                else:
                    max_expr = self.builder_.emit(R.max(abs_expr, axis=-2))
                if expr.struct_info.ndim > 2:
                    max_expr = self.builder_.emit(R.squeeze(max_expr))
                return max_expr

        return ParamsAndOutputsMutator(mod, self.op_mode).transform()


@tvm.transform.module_pass(opt_level=0, name="SmoothQuantOpConverter")
class SmoothQuantOpConverter:
    def __init__(self, op_mode: str) -> None:
        self.op_name = "quantize" if op_mode in QSCHEMES else op_mode

    def transform_module(self, mod: tvm.IRModule, ctx: tvm.transform.PassContext) -> tvm.IRModule:
        attrs = {"mode": "identity"}
        data = wildcard()
        scale = wildcard()
        pattern = is_op("relax.annotate.smooth")(data, scale).has_attr(attrs)

        def rewriter(_, matchings):
            kind = matchings[pattern].attrs.kind
            return R.smooth(matchings[data], matchings[scale], kind=kind, mode=self.op_name)

        new_mod = tvm.IRModule()
        for gv, func in mod.functions.items():
            if isinstance(func, relax.Function):
                new_mod[gv] = rewrite_call(pattern, rewriter, func)
            else:
                new_mod[gv] = func
        return new_mod


@tvm.transform.module_pass(opt_level=0, name="SmoothQuantLegalizer")
class SmoothQuantLegalizer:
    """
    Pass that converts matmul(fp16, fp16) -> quantize + matmul(int8, int8) + dequantize.
    """
    def __init__(self, op_mode: str, adtype: str = "int8", wdtype: str = "int8"):
        self.dtype_act = adtype
        self.dtype_weight = wdtype
        # Operation mode of the legalizer: "smoothing" or "quantization"
        assert op_mode in OPMODES, f"unsupported operation mode: '{op_mode}'"
        self.op_mode = op_mode

    def transform_module(self, mod: tvm.IRModule, ctx: tvm.transform.PassContext) -> tvm.IRModule:
        attrs = {"mode": "quantize"}
        act_scale = wildcard()
        w_scale = wildcard()
        lhs_sm = is_op("relax.annotate.smooth")(wildcard(), act_scale).has_attr(attrs)
        rhs_sm = is_op("relax.annotate.smooth")(wildcard(), w_scale).has_attr(attrs)
        permute = is_op("relax.permute_dims")(rhs_sm)
        pattern = is_op("relax.matmul")(lhs_sm, permute)

        def rewriter(_, matchings):
            def _make_quantize(call: tvm.relax.Call, out_dtype: str):
                in_dtype = call.args[0].struct_info.dtype
                min_value = tvm.tir.min_value(out_dtype).astype(in_dtype)
                max_value = tvm.tir.max_value(out_dtype).astype(in_dtype)
                scale = call.args[1]
                if self.op_mode == "smq_q8i8f16_1":
                    scale = R.expand_dims(scale, axis=[1])
                data = R.round(R.divide(call.args[0], scale))
                return R.astype(R.clip(data, min_value, max_value), dtype=out_dtype)
            
            def _make_dequantize(
                call: tvm.relax.Call,
                scale1: tvm.relax.Constant,
                scale2: tvm.relax.Constant,
                out_dtype: str,
            ):
                if out_dtype == "float32":
                    return R.multiply(R.astype(call, dtype=out_dtype), R.multiply(scale1, scale2))
                else:
                    assert out_dtype == "float16"
                    min_value = tvm.tir.min_value(out_dtype).astype("float32")
                    max_value = tvm.tir.max_value(out_dtype).astype("float32")
                    dq_scale = R.multiply(R.astype(scale1, "float32"), R.astype(scale2, "float32"))
                    out = R.multiply(R.astype(call, dtype="float32"), dq_scale)
                    return R.astype(R.clip(out, min_value, max_value), dtype=out_dtype)

            lhs = _make_quantize(matchings[lhs_sm], self.dtype_act)
            rhs = _make_quantize(matchings[rhs_sm], self.dtype_weight)
            mm = R.linear(lhs, rhs, out_dtype="int32")
            dtype = matchings[pattern].struct_info.dtype
            return _make_dequantize(mm, matchings[act_scale], matchings[w_scale], dtype)

        new_mod = tvm.IRModule()
        for gv, func in mod.functions.items():
            if isinstance(func, relax.Function):
                new_mod[gv] = rewrite_call(pattern, rewriter, func)
            else:
                new_mod[gv] = func
        return new_mod
