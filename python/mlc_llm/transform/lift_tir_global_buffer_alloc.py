"""Lift global buffer allocation in TIR to graph level"""

from typing import Dict, List, Tuple

import tvm
from tvm import relax, tir
from tvm.ir.module import IRModule
from tvm.relax.analysis import remove_all_unused
from tvm.relax.expr_functor import PyExprMutator, mutator


def remove_global_buf_alloc(
    func: tir.PrimFunc,
) -> Tuple[tir.PrimFunc, List[relax.TensorStructInfo]]:
    """Remove the global buffer allocation for a given TIR PrimFunc."""
    assert isinstance(func.body, tir.BlockRealize)
    params = list(func.params)
    buffer_map = dict(func.buffer_map)
    tensor_sinfo = []
    alloc_buffers = []

    insertion_point = len(params)
    while params[insertion_point - 1].dtype != "handle":
        insertion_point -= 1
        assert insertion_point >= 1

    prev_root_block = func.body.block
    for buf_alloc in func.body.block.alloc_buffers:
        if buf_alloc.scope() == "global":
            param = tir.Var("var_" + buf_alloc.name, "handle")
            params.insert(insertion_point, param)
            insertion_point += 1
            buffer_map[param] = buf_alloc
            tensor_sinfo.append(
                relax.TensorStructInfo(buf_alloc.shape, buf_alloc.dtype)
            )
        else:
            alloc_buffers.append(buf_alloc)

    if len(tensor_sinfo) == 0:
        return func, []

    assert len(prev_root_block.iter_vars) == 0
    assert len(prev_root_block.reads) == 0
    assert len(prev_root_block.writes) == 0
    assert len(prev_root_block.match_buffers) == 0
    assert prev_root_block.name_hint == "root"
    assert prev_root_block.init is None
    root_block = tir.Block(
        iter_vars=[],
        reads=[],
        writes=[],
        name_hint="root",
        body=prev_root_block.body,
        alloc_buffers=alloc_buffers,
        annotations=prev_root_block.annotations,
    )

    updated_func = tir.PrimFunc(
        params=params,
        body=tir.BlockRealize(iter_values=[], predicate=True, block=root_block),
        ret_type=func.ret_type,
        buffer_map=buffer_map,
        attrs=func.attrs,
    )
    return updated_func, tensor_sinfo


def contain_symbolic_var(tensor_sinfo: relax.TensorStructInfo) -> bool:
    assert isinstance(tensor_sinfo.shape, relax.ShapeExpr)
    for v in tensor_sinfo.shape.values:
        if not isinstance(v, tir.IntImm):
            return True
    return False


def resolve_tir_var_mapping(
    func: tir.PrimFunc, call: relax.Call, tensor_sinfo: List[relax.TensorStructInfo]
) -> Tuple[List[relax.TensorStructInfo], bool]:
    """Resolve the TIR symbolic var relationship across sides of PrimFunc and Relax Function"""
    var_map: Dict[tir.Var, tir.PrimExpr] = dict()

    n_arg = len(call.args[1].fields)
    for i in range(n_arg):
        buffer_shape = func.buffer_map[func.params[i]].shape
        arg_shape = call.args[1][i].struct_info.shape.values
        assert len(buffer_shape) == len(arg_shape)
        for vl, vr in zip(buffer_shape, arg_shape):
            if isinstance(vl, tir.Var):
                var_map[vl] = vr
            elif not isinstance(vl, tir.IntImm):
                return [], False

    ret_tensors = call.sinfo_args[0]
    ret_tensors = (
        [ret_tensors]
        if isinstance(ret_tensors, relax.TensorStructInfo)
        else list(ret_tensors.fields)
    )
    for i in range(len(ret_tensors)):
        buffer_shape = func.buffer_map[func.params[n_arg + i]].shape
        ret_tensor_shape = ret_tensors[i].shape.values
        assert len(buffer_shape) == len(ret_tensor_shape)
        for vl, vr in zip(buffer_shape, ret_tensor_shape):
            if isinstance(vl, tir.Var):
                var_map[vl] = vr
            elif not isinstance(vl, tir.IntImm):
                return [], False

    updated_tensor_sinfo = []
    for sinfo in tensor_sinfo:
        if not contain_symbolic_var(sinfo):
            updated_tensor_sinfo.append(sinfo)
            continue

        new_shape = []
        for v in sinfo.shape.values:
            new_shape.append(tir.stmt_functor.substitute(v, var_map))
        updated_tensor_sinfo.append(relax.TensorStructInfo(new_shape, sinfo.dtype))
    return updated_tensor_sinfo, True


@tvm.transform.module_pass(opt_level=0, name="LiftTIRGlobalBufferAlloc")
class LiftTIRGlobalBufferAlloc:
    def transform_module(
        self, mod: IRModule, ctx: tvm.transform.PassContext
    ) -> IRModule:
        @mutator
        class TIRGlobalAllocRewriter(PyExprMutator):
            def __init__(self, mod: IRModule):
                super().__init__(mod)
                self.mod = mod
                self.gv2new_tensor_sinfo: Dict[
                    tvm.ir.GlobalVar, Tuple[List[relax.TensorStructInfo], tir.PrimFunc]
                ] = dict()

            def transform(self) -> IRModule:
                for gv, func in self.mod.functions.items():
                    if isinstance(func, tir.PrimFunc):
                        updated_func, tensor_sinfo_list = remove_global_buf_alloc(func)
                        if len(tensor_sinfo_list) > 0:
                            self.gv2new_tensor_sinfo[gv] = (tensor_sinfo_list, func)
                            self.builder_.update_func(gv, updated_func)

                self.mod = self.builder_.get()
                for gv, func in self.mod.functions.items():
                    if not isinstance(func, relax.Function):
                        continue
                    updated_func = self.visit_expr(func)
                    updated_func = remove_all_unused(updated_func)
                    self.builder_.update_func(gv, updated_func)
                return self.builder_.get()

            def visit_call_(self, call: relax.Call):
                call = self.visit_expr_post_order(call)
                if (
                    call.op != tvm.ir.Op.get("relax.call_tir")
                    or call.args[0] not in self.gv2new_tensor_sinfo
                ):
                    return call

                gv = call.args[0]
                tensor_sinfo, func_before_update = self.gv2new_tensor_sinfo[gv]

                assert len(call.sinfo_args) == 1
                if any(contain_symbolic_var(sinfo) for sinfo in tensor_sinfo):
                    tensor_sinfo, success = resolve_tir_var_mapping(
                        func_before_update, call, tensor_sinfo
                    )
                    if not success:
                        # Cannot resolve TIR var mapping. Fall back to no lifting.
                        self.builder_.update_func(gv, func_before_update)
                        self.gv2new_tensor_sinfo.pop(gv)
                        return call

                if isinstance(call.sinfo_args[0], relax.TensorStructInfo):
                    new_call = relax.Call(
                        call.op,
                        args=call.args,
                        sinfo_args=[
                            relax.TupleStructInfo(list(call.sinfo_args) + tensor_sinfo)
                        ],
                        attrs=call.attrs,
                    )
                    emitted_tuple = self.builder_.emit(new_call)
                    return relax.TupleGetItem(emitted_tuple, 0)
                else:
                    assert isinstance(call.sinfo_args[0], relax.TupleStructInfo)
                    return relax.Call(
                        call.op,
                        args=call.args,
                        sinfo_args=[
                            relax.TupleStructInfo(
                                list(call.sinfo_args[0].fields) + tensor_sinfo
                            )
                        ],
                        attrs=call.attrs,
                    )

        return TIRGlobalAllocRewriter(mod).transform()
