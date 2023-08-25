import tvm
from tvm import IRModule
from tvm import relax
from tvm.relax.expr import Expr, Function
from tvm.script import relax as R
from tvm.script import tir as T
from tvm.relax.analysis import remove_all_unused
from tvm.relax.op.builtin import stop_lift_params
from tvm.ir import Op
import functools
from typing import Dict, Iterable, Tuple, Callable, List


@relax.expr_functor.mutator
class MutatorFusedFunc(tvm.relax.PyExprMutator):
    def __init__(self, map) -> None:
        super().__init__()
        self.map = map

    def visit_var_def_(self, var: relax.Var) -> relax.Var:
        if var in self.map:
            new_var = relax.Var(var.name_hint, self.map[var])
            return new_var
        return var

    def visit_call_(self, call_node: relax.Call) -> relax.Call:
        if not isinstance(call_node.op, tvm.relax.expr.Var):
            return super().visit_call_(call_node)
        new_args = [self.visit_expr(arg) for arg in call_node.args]
        local_func = self.lookup_binding(call_node.op)
        local_var_binding = {
            param: argument.struct_info for param, argument in zip(local_func.params, new_args)
        }
        new_local_func = MutatorFusedFunc(local_var_binding).visit_expr(local_func)
        return new_local_func(*new_args)


@relax.expr_functor.mutator
class FindMatMul(tvm.relax.PyExprMutator):
    def __init__(self, mod: tvm.IRModule, args) -> None:
        super().__init__(mod)
        self.mod_ = mod
        self.matches_ = {}
        self.global_var = None
        self.mapping = {}
        self._process_group = None
        self.args = args

    def _init_process_group(self):
        if self._process_group is None:
            # world_size = tvm.tir.SizeVar("world_size", "int64")
            rank = tvm.tir.SizeVar("rank", "int64")
            sinfo = relax.ShapeStructInfo([rank])
            self._process_group = relax.Var("process_group", sinfo)
        return self._process_group

    def visit_function_(self, node):
        assert self._process_group is None
        node = super().visit_function_(node)
        if self._process_group is not None:
            node = relax.Function(
                params=[*node.params, self._process_group],
                body=node.body,
                attrs=node.attrs,
                ret_struct_info=node.struct_info,
            )
            self._process_group = None
        return node

    def transform(self):
        mod_first_mutatot = {}

        for global_var, func in self.mod_.functions.items():
            if isinstance(func, relax.Function):
                self.global_var = global_var
                func = self.visit_expr(func)
            mod_first_mutatot[global_var] = func
        mod = tvm.IRModule(mod_first_mutatot)
        # Update the IRModule to have the new callers
        mod_second_mutatot = {}
        mutate_fused_func_shapes = MutatorFusedFunc(self.mapping)
        for global_var, func in mod.functions.items():
            if isinstance(func, relax.Function):
                # Update the local primitive function call nodes
                func = mutate_fused_func_shapes.visit_expr(func)
                func = self.builder_.normalize(func)
                func = remove_all_unused(func)
                func = self.builder_.normalize(func)

            mod_second_mutatot[global_var] = func
        return tvm.IRModule(mod_second_mutatot)

    def visit_call_(self, call_node: relax.Call) -> relax.Call:
        is_fused_matmul = "matmul" in str(call_node.op) and isinstance(
            call_node.op, tvm.ir.expr.GlobalVar
        )
        is_fused_linear = "fused_relax_permute_dims_relax_matmul" in str(call_node.op)

        if is_fused_matmul:
            process_group = self._init_process_group()
            # rank = process_group.struct_info.values[0]
            rank = process_group.struct_info.values[0]
            # rank = 0
            world_size = self.args.num_gpus

            if is_fused_linear:
                axes_to_slice = 0
                weights, activation, *bias = call_node.args

                if len(bias) > 1:
                    print("Not Implemented")
                    assert False

                if len(weights.struct_info.shape) != 2:
                    return call_node
                outfeatures, infeatures = weights.struct_info.shape
                sharded_weights_shape = (outfeatures // world_size, infeatures)
                sharded_weights_s_info = relax.TensorStructInfo(
                    sharded_weights_shape, dtype=weights.struct_info.dtype
                )
                matmul_shape = [
                    *activation.struct_info.shape.values[:-1],
                    weights.struct_info.shape[0] // world_size,
                ]
                matmul_s_info = relax.TensorStructInfo(
                    matmul_shape, dtype=activation.struct_info.dtype
                )

                if bias:
                    infeature_bias, batch_bias, outfeatures_bias = bias[0].struct_info.shape
                    sharded_bias_shape = (
                        infeature_bias,
                        batch_bias,
                        outfeatures_bias // world_size,
                    )
                    sharded_bias_s_info = relax.TensorStructInfo(
                        sharded_bias_shape, dtype=bias[0].struct_info.dtype
                    )
                    output_struct_info = relax.FuncStructInfo(
                        (
                            sharded_weights_s_info,
                            activation.struct_info,
                            sharded_bias_s_info,
                        ),
                        matmul_s_info,
                        True,
                    )
                    bias_param = self.mod_[call_node.op].params[2]
                    self.mapping[bias_param] = sharded_bias_s_info
                else:
                    output_struct_info = relax.FuncStructInfo(
                        (sharded_weights_s_info, activation.struct_info),
                        matmul_s_info,
                        True,
                    )

            else:
                axes_to_slice = 1
                activation, weights, *bias = call_node.args
                if bias:
                    print(
                        "Not Implemented. Matmul (if it's not linear) with Bias is not implemented"
                    )
                    assert False
                if len(weights.struct_info.shape) != 2:
                    return call_node
                infeatures, outfeatures = weights.struct_info.shape
                sharded_weights_shape = (infeatures, outfeatures // world_size)
                sharded_weights_s_info = relax.TensorStructInfo(
                    sharded_weights_shape, dtype=weights.struct_info.dtype
                )
                matmul_shape = [
                    *activation.struct_info.shape.values[:-1],
                    weights.struct_info.shape[-1] // world_size,
                ]
                matmul_s_info = relax.TensorStructInfo(
                    matmul_shape, dtype=activation.struct_info.dtype
                )
                output_struct_info = relax.FuncStructInfo(
                    (activation.struct_info, sharded_weights_s_info),
                    matmul_s_info,
                    True,
                )
            all_gather_s_info = call_node.struct_info

            weights_param = self.mod_[call_node.op].params[axes_to_slice]
            self.mapping[weights_param] = sharded_weights_s_info

            output_param = self.mod_[call_node.op].body.blocks[0].bindings[1].var
            self.mapping[output_param] = matmul_s_info

            tvm.relax.expr._update_struct_info(call_node.op, output_struct_info)

            self.matches_[call_node.op] = call_node.op
            sharded_weights = tvm.relax.op.strided_slice(
                weights,
                axes=[axes_to_slice],
                begin=[(rank * outfeatures) // world_size],
                end=[((rank + 1) * outfeatures) // world_size],
            )
            sharded_weights = stop_lift_params(sharded_weights)
            if is_fused_linear:
                if bias:
                    sharded_bias = tvm.relax.op.strided_slice(
                        bias[0],
                        axes=[2],
                        begin=[(rank * outfeatures_bias) // world_size],
                        end=[((rank + 1) * outfeatures_bias) // world_size],
                        assume_inbound=True,
                    )
                    sharded_bias = stop_lift_params(sharded_bias)
                    new_matmul = call_node.op(sharded_weights, activation, sharded_bias)
                else:
                    new_matmul = call_node.op(sharded_weights, activation)

            else:
                new_matmul = call_node.op(activation, sharded_weights)

            args = relax.expr.Tuple([new_matmul])
            output = relax.op.call_dps_packed(
                "tvm.torch.distributed.collective.allgather", args, all_gather_s_info
            )

            return output

        return call_node


@tvm.ir.transform.module_pass(opt_level=0, name="PassPostFusionMultiGPU")
class PassPostFusionMultiGPU:
    def __init__(self, args):
        self.args = args

    def transform_module(self, mod: IRModule, ctx: tvm.transform.PassContext) -> IRModule:
        return FindMatMul(mod, self.args).transform()
