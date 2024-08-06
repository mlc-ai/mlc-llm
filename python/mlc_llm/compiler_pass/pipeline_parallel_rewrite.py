"""A compiler pass that rewrites IR for pipeline parallelism."""

from typing import Dict, List, Optional, Tuple

import tvm
from tvm import relax, tir
from tvm.ir.module import IRModule
from tvm.relax.expr_functor import PyExprMutator, PyExprVisitor, mutator, visitor


@tvm.transform.module_pass(opt_level=0, name="PipelineParallelRewrite")
class PipelineParallelRewrite:  # pylint: disable=too-few-public-methods
    """A compiler pass that rewrites IR for pipeline parallelism."""

    def transform_module(
        self,
        mod: IRModule,
        _ctx: tvm.transform.PassContext,
    ) -> IRModule:
        """IRModule-level transformation"""
        return _PipelineParallelRewriter(mod.clone()).transform()


@mutator
class _PipelineParallelRewriter(PyExprMutator):  # pylint: disable=abstract-method
    def __init__(self, mod: IRModule):
        super().__init__(mod)
        self.mod = mod
        self.old_packed_params_var: relax.Var
        self.new_main_packed_params_var: relax.Var
        self.new_stage_func_packed_params: relax.Var
        self.undefined_shape_vars_remap: Dict[tir.Var, tir.Var]
        self.undefined_param_shape_vars_remap: Dict[tir.Var, tir.Var]

    def transform(self) -> IRModule:  # pylint: disable=too-many-locals
        """Entry point of the transformation"""
        for g_var, func in self.mod.functions_items():
            if not isinstance(func, relax.Function) or "pipeline_parallel_stages" not in func.attrs:
                continue
            num_stages = int(func.attrs["pipeline_parallel_stages"])
            if num_stages == 1:
                continue

            pipeline_stages, stage_send_vars, stage_receive_vars = _extract_pipeline_stages(func)
            assert len(pipeline_stages) == num_stages, (
                "Number of pipeline stages mismatches: "
                f"expecting {num_stages} stages, but {len(pipeline_stages)} are found in the IR."
            )

            required_func_params = _analyze_required_func_params(pipeline_stages, func.params)

            assert "num_input" in func.attrs
            num_input = int(func.attrs["num_input"])
            assert (
                len(func.params) == num_input + 1
                and isinstance(func.params[num_input], relax.Var)
                and func.params[num_input].name_hint == "packed_params"
            ), 'Only the extra "packed_params" parameter is allowed'
            self.old_packed_params_var = func.params[num_input]
            self.new_main_packed_params_var = relax.Var("packed_params", relax.ObjectStructInfo())
            for required_params in required_func_params:
                for i, param in enumerate(required_params):
                    if param.same_as(self.old_packed_params_var):
                        required_params.pop(i)
                        break
            func_output = func.body.body
            assert isinstance(func_output, relax.Var)

            stage_func_gvs = []
            caller_args_list = []
            for i in range(num_stages):
                stage_func_gv, caller_args = self._create_stage_func(
                    g_var.name_hint + f"_stage{i}",
                    pipeline_stages[i],
                    required_func_params[i],
                    stage_receive_vars[i],
                    stage_send_vars[i],
                    func.attrs,
                    func_output=func_output if i == num_stages - 1 else None,
                )
                stage_func_gvs.append(stage_func_gv)
                caller_args_list.append(caller_args)

            # Create and update the entry function, which dispatches toz the stage functions
            # according to the disco worker group id.
            bb = relax.BlockBuilder()
            params = list(func.params[:-1]) + [self.new_main_packed_params_var]
            with bb.function(g_var.name_hint, params=params):
                dispatch_func_args = []
                for stage_func_gv, caller_args in zip(stage_func_gvs, caller_args_list):
                    dispatch_func_args.append([stage_func_gv] + caller_args)
                output = bb.emit(
                    relax.op.call_builtin_with_ctx(
                        "mlc.multi_gpu.DispatchFunctionByGroup",
                        args=[dispatch_func_args],
                        sinfo_args=relax.ObjectStructInfo(),
                    )
                )
                dispatch_func_gv = bb.emit_func_output(output)
            dispatch_func = bb.finalize()[dispatch_func_gv]
            self.builder_.update_func(g_var, dispatch_func)

        return self.builder_.finalize()

    def _create_stage_func(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        func_name: str,
        stage_bindings: List[relax.Binding],
        required_func_params: List[relax.Var],
        stage_receive_vars: List[relax.Var],
        stage_send_vars: List[relax.Var],
        func_attrs: tvm.ir.DictAttrs,
        func_output: Optional[relax.Var],
    ) -> Tuple[tvm.ir.GlobalVar, List[relax.Expr]]:
        self.undefined_shape_vars_remap = {}
        self.undefined_param_shape_vars_remap = {}

        # Prepare the func parameters (except the shape variables and packed params)
        params, args = self._prepare_stage_func_params_and_args(required_func_params)
        for new_param, old_param in zip(params, required_func_params):
            self.set_var_remap(old_param.vid, new_param)
        # Create new packed params
        self.new_stage_func_packed_params = relax.Var("packed_params", relax.ObjectStructInfo())
        self.set_var_remap(self.old_packed_params_var.vid, self.new_stage_func_packed_params)

        new_func_outputs = []
        with self.builder_.function(func_name, pure=False):
            with self.builder_.dataflow():
                # Emit the tensors received from last stage.
                for receive_var in stage_receive_vars:
                    new_receive_var = self.builder_.emit(
                        relax.call_dps_packed(
                            "runtime.disco.recv_from_prev_group",
                            args=[],
                            out_sinfo=self._update_struct_info(receive_var.struct_info),
                        ),
                        name_hint=receive_var.name_hint,
                    )
                    self.set_var_remap(receive_var.vid, new_receive_var)
                # Process the bindings in this stage.
                for stage_binding in stage_bindings:
                    if stage_binding.var in stage_send_vars or stage_binding.var.same_as(
                        func_output
                    ):
                        assert isinstance(stage_binding, relax.VarBinding)
                        new_var = self.builder_.emit_output(
                            self.visit_expr(stage_binding.value),
                            name_hint=stage_binding.var.name_hint,
                        )
                        self.set_var_remap(stage_binding.var.vid, new_var)
                        new_func_outputs.append(new_var)
                    else:
                        self.visit_binding(stage_binding)
            # Emit the calls to send tensors to the next stage.
            for send_var in stage_send_vars:
                new_send_var = self.get_var_remap(send_var.vid)
                self.builder_.emit(
                    relax.Call(
                        relax.ExternFunc("runtime.disco.send_to_next_group"),
                        args=[new_send_var],
                        sinfo_args=None,
                    )
                )
            # Create the param for the shape variables.
            shape_var_params = []
            shape_var_args = []
            for shape_var_arg, shape_var_param in self.undefined_shape_vars_remap.items():
                if shape_var_arg not in self.undefined_param_shape_vars_remap:
                    shape_var_params.append(shape_var_param)
                    shape_var_args.append(shape_var_arg)
            params.append(relax.Var("s", relax.ShapeStructInfo(shape_var_params)))
            args.append(relax.ShapeExpr(shape_var_args))
            # Add the packed params.
            params.append(self.new_stage_func_packed_params)
            args.append(self.new_main_packed_params_var)
            # Conclude the function.
            if func_output is not None:
                assert len(new_func_outputs) == 1
            new_gv = self.builder_.emit_func_output(
                (
                    new_func_outputs[0]
                    if len(new_func_outputs) == 1
                    and isinstance(new_func_outputs[0].struct_info, relax.TupleStructInfo)
                    else new_func_outputs
                ),
                params=params,
            )

        new_func = (
            self.builder_.get()[new_gv]
            .with_attrs(func_attrs)
            .with_attr("num_input", len(params) - 1)
            .without_attr("global_symbol")
            .without_attr("pipeline_parallel_stages")
        )
        self.builder_.update_func(new_gv, new_func)
        return new_gv, args

    def visit_var_binding_(self, binding: relax.VarBinding) -> None:
        if not isinstance(binding.value, relax.TupleGetItem):
            super().visit_var_binding_(binding)
            return

        tuple_value = self.visit_expr(binding.value.tuple_value)
        if not tuple_value.same_as(self.new_stage_func_packed_params):
            super().visit_var_binding_(binding)
            return

        assert isinstance(binding.var.struct_info, relax.TensorStructInfo)
        cur_num_undefined_param_shape_vars = len(self.undefined_param_shape_vars_remap)
        new_tensor_struct_info = self._update_struct_info(
            binding.var.struct_info, self.undefined_param_shape_vars_remap
        )
        has_new_undefined_shape_var = (
            len(self.undefined_param_shape_vars_remap) != cur_num_undefined_param_shape_vars
        )
        self.undefined_shape_vars_remap = {
            **self.undefined_shape_vars_remap,
            **self.undefined_param_shape_vars_remap,
        }
        ret_sinfo = (
            new_tensor_struct_info if not has_new_undefined_shape_var else relax.ObjectStructInfo()
        )
        call = relax.call_pure_packed(
            "vm.builtin.tuple_getitem",
            self.new_stage_func_packed_params,
            relax.PrimValue(binding.value.index),
            sinfo_args=ret_sinfo,
        )
        new_binding_var = self.builder_.emit(call, binding.var.name_hint)
        if has_new_undefined_shape_var:
            new_binding_var = self.builder_.match_cast(
                new_binding_var, new_tensor_struct_info, binding.var.name_hint + "_cast"
            )
        self.set_var_remap(binding.var.vid, new_binding_var)

    def visit_call_(self, call: relax.Call) -> relax.Call:  # pylint: disable=arguments-renamed
        call = super().visit_call_(call)
        return relax.Call(
            call.op,
            call.args,
            call.attrs,
            sinfo_args=[self._update_struct_info(struct_info) for struct_info in call.sinfo_args],
        )

    def _prepare_stage_func_params_and_args(
        self, required_func_params: List[relax.Var]
    ) -> Tuple[List[relax.Var], List[relax.Expr]]:
        params: List[relax.Var] = []
        args: List[relax.Expr] = []
        for required_param in required_func_params:
            struct_info = self._update_struct_info(required_param.struct_info)
            params.append(relax.Var(required_param.name_hint, struct_info))
            args.append(required_param)

        return params, args

    def _update_struct_info(
        self,
        struct_info: relax.StructInfo,
        undefined_var_remap: Optional[Dict[tir.Var, tir.Var]] = None,
    ) -> relax.StructInfo:
        if undefined_var_remap is None:
            undefined_var_remap = self.undefined_shape_vars_remap
        if isinstance(struct_info, relax.TensorStructInfo):
            return (
                relax.TensorStructInfo(
                    self._update_shape(struct_info.shape.values, undefined_var_remap),
                    struct_info.dtype,
                )
                if struct_info.shape is not None and isinstance(struct_info.shape, relax.ShapeExpr)
                else struct_info
            )
        if isinstance(struct_info, relax.ShapeStructInfo):
            return (
                relax.ShapeStructInfo(self._update_shape(struct_info.values, undefined_var_remap))
                if struct_info.values is not None
                else struct_info
            )
        if isinstance(struct_info, relax.ObjectStructInfo):
            return relax.ObjectStructInfo()
        if isinstance(struct_info, relax.TupleStructInfo):
            return relax.TupleStructInfo(
                [self._update_struct_info(field_sinfo) for field_sinfo in struct_info.fields]
            )
        return struct_info

    def _copy_undefined_var(
        self, expr: tir.PrimExpr, undefined_var_remap: Dict[tir.Var, tir.Var]
    ) -> None:
        def _visit_expr(e: tir.PrimExpr) -> None:
            if isinstance(e, tir.Var) and e not in undefined_var_remap:
                new_var = tir.Var(e.name, e.dtype)
                undefined_var_remap[e] = new_var

        tir.stmt_functor.post_order_visit(expr, _visit_expr)

    def _update_shape(
        self, shape: List[tir.PrimExpr], undefined_var_remap: Dict[tir.Var, tir.Var]
    ) -> List[tir.PrimExpr]:
        new_shape = []
        for v in shape:
            self._copy_undefined_var(v, undefined_var_remap)
            new_shape.append(tir.stmt_functor.substitute(v, undefined_var_remap))
        return new_shape


def _extract_pipeline_stages(
    func: relax.Function,
) -> Tuple[List[List[relax.Binding]], List[List[relax.Var]], List[List[relax.Var]]]:
    pipeline_stages: List[List[relax.Binding]] = []
    stage_send_vars: List[List[relax.Var]] = []
    stage_receive_vars: List[List[relax.Var]] = []

    # Requiring that the function has only one body block which is a dataflow block
    assert isinstance(func.body, relax.SeqExpr)
    assert len(func.body.blocks) == 1
    assert isinstance(func.body.blocks[0], relax.DataflowBlock)
    bindings = func.body.blocks[0].bindings

    boundary_var = None
    current_stage_bindings: List[relax.Binding] = []
    current_stage_receive_vars: List[relax.Var] = []
    for binding in bindings:
        if (
            isinstance(binding, relax.VarBinding)
            and isinstance(binding.value, relax.Call)
            and binding.value.op == tvm.ir.Op.get("relax.call_pure_packed")
            and binding.value.args[0].global_symbol == "mlc.pipeline_parallel_stage_boundary"
        ):
            assert len(current_stage_bindings) > 0
            pipeline_stages.append(current_stage_bindings)
            assert all(receive_var is not None for receive_var in current_stage_receive_vars)
            stage_receive_vars.append(current_stage_receive_vars)
            args = binding.value.args[1:]
            assert len(args) >= 1 and all(isinstance(arg, relax.Var) for arg in args)
            stage_send_vars.append(list(args))

            boundary_var = binding.var
            current_stage_bindings = []
            current_stage_receive_vars = [boundary_var] if len(args) == 1 else [None for _ in args]
        elif (
            isinstance(binding, relax.VarBinding)
            and isinstance(binding.value, relax.TupleGetItem)
            and binding.value.tuple_value.same_as(boundary_var)
        ):
            current_stage_receive_vars[binding.value.index] = binding.var
        else:
            current_stage_bindings.append(binding)

    assert len(current_stage_bindings) > 0
    pipeline_stages.append(current_stage_bindings)
    assert all(receive_var is not None for receive_var in current_stage_receive_vars)
    stage_receive_vars.append(current_stage_receive_vars)
    stage_send_vars.append([])

    return pipeline_stages, stage_send_vars, stage_receive_vars


def _analyze_required_func_params(
    pipeline_stages: List[List[relax.Binding]], func_params: List[relax.Var]
) -> List[List[relax.Var]]:
    analyzer = _RequiredFuncParamAnalyzer(func_params)
    required_func_params: List[List[relax.Var]] = []
    for stage_bindings in pipeline_stages:
        required_params: List[relax.Var]
        required_params = analyzer.run(stage_bindings)
        required_func_params.append(required_params)
    return required_func_params


@visitor
class _RequiredFuncParamAnalyzer(PyExprVisitor):
    """The IR visitor which analyzes the required func parameters in each pipeline stage."""

    def __init__(self, func_params: List[relax.Var]) -> None:
        self.func_params = set(func_params)
        self.required_params: List[relax.Var]

    def run(self, stage_bindings: List[relax.Binding]) -> List[relax.Var]:
        """Entry point of the visitor."""
        self.required_params = []
        for binding in stage_bindings:
            self.visit_binding(binding)
        return self.required_params

    def visit_var_(self, var: relax.Var) -> None:  # pylint: disable=arguments-renamed
        if var in self.func_params:
            if var not in self.required_params:
                self.required_params.append(var)
