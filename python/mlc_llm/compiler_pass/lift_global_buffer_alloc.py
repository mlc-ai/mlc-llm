"""A compiler pass that lifts TIR-level global allocation to Relax."""

from typing import Dict, List, Tuple

import tvm
from tvm import relax, tir
from tvm.ir.module import IRModule
from tvm.relax.analysis import remove_all_unused
from tvm.relax.expr_functor import PyExprMutator, mutator


@tvm.transform.module_pass(opt_level=0, name="LiftTIRGlobalBufferAlloc")
class LiftTIRGlobalBufferAlloc:  # pylint: disable=too-few-public-methods
    """A compiler pass that lifts TIR-level global allocation to Relax."""

    def transform_module(
        self,
        mod: IRModule,
        _ctx: tvm.transform.PassContext,
    ) -> IRModule:
        """IRModule-level transformation"""
        return _TIRGlobalAllocRewriter(mod).transform()


@mutator
class _TIRGlobalAllocRewriter(PyExprMutator):  # pylint: disable=abstract-method
    def __init__(self, mod: IRModule):
        super().__init__(mod)
        self.mod = mod
        self.gv2new_tensor_sinfo: Dict[
            tvm.ir.GlobalVar, Tuple[tvm.ir.GlobalVar, List[relax.TensorStructInfo], tir.PrimFunc]
        ] = {}

    def transform(self) -> IRModule:
        """Entry point of the transformation"""
        for g_var, func in self.mod.functions_items():
            if isinstance(func, tir.PrimFunc):
                updated_func, tensor_sinfo_list = remove_global_buf_alloc(func)
                if len(tensor_sinfo_list) > 0:
                    new_gv = self.builder_.add_func(updated_func, g_var.name_hint)
                    self.gv2new_tensor_sinfo[g_var] = (new_gv, tensor_sinfo_list, func)

        self.mod = self.builder_.get()
        for g_var, func in self.mod.functions_items():
            if isinstance(func, relax.Function):
                updated_func = self.visit_expr(func)
                updated_func = remove_all_unused(updated_func)
                self.builder_.update_func(g_var, updated_func)

        mod = self.builder_.get()
        return relax.transform.DeadCodeElimination()(mod)

    def visit_call_(self, call: relax.Call):  # pylint: disable=arguments-renamed
        call = self.visit_expr_post_order(call)
        if (
            call.op != tvm.ir.Op.get("relax.call_tir")
            or call.args[0] not in self.gv2new_tensor_sinfo
        ):
            return call

        g_var = call.args[0]
        new_gv, tensor_sinfo, func_before_update = self.gv2new_tensor_sinfo[g_var]

        assert len(call.sinfo_args) == 1
        if any(_has_symbolic_var(sinfo) for sinfo in tensor_sinfo):
            tensor_sinfo, success = _resolve_tir_var_mapping(func_before_update, call, tensor_sinfo)
            if not success:
                # Cannot resolve TIR var mapping. Fall back to no lifting.
                self.gv2new_tensor_sinfo.pop(g_var)
                return call

        args = list(call.args)
        args[0] = new_gv
        if isinstance(call.sinfo_args[0], relax.TensorStructInfo):
            new_call = relax.Call(
                call.op,
                args=args,
                sinfo_args=[relax.TupleStructInfo(list(call.sinfo_args) + tensor_sinfo)],
                attrs=call.attrs,
            )
            emitted_tuple = self.builder_.emit(new_call)
            return relax.TupleGetItem(emitted_tuple, 0)
        assert isinstance(call.sinfo_args[0], relax.TupleStructInfo)
        return relax.Call(
            call.op,
            args=args,
            sinfo_args=[relax.TupleStructInfo(list(call.sinfo_args[0].fields) + tensor_sinfo)],
            attrs=call.attrs,
        )


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
            tensor_sinfo.append(relax.TensorStructInfo(buf_alloc.shape, buf_alloc.dtype))
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


def _has_symbolic_var(tensor_sinfo: relax.TensorStructInfo) -> bool:
    assert isinstance(tensor_sinfo.shape, relax.ShapeExpr)
    for dim in tensor_sinfo.shape.values:
        if not isinstance(dim, tir.IntImm):
            return True
    return False


def _resolve_tir_var_mapping(  # pylint: disable=too-many-locals
    func: tir.PrimFunc,
    call: relax.Call,
    tensor_sinfo: List[relax.TensorStructInfo],
) -> Tuple[List[relax.TensorStructInfo], bool]:
    """Resolve the TIR symbolic var relationship across sides of PrimFunc and Relax Function"""
    var_map: Dict[tir.Var, tir.PrimExpr] = {}

    n_arg = len(call.args[1].fields)
    for i in range(n_arg):
        buffer_shape = func.buffer_map[func.params[i]].shape
        arg_shape = call.args[1][i].struct_info.shape.values
        assert len(buffer_shape) == len(arg_shape)
        for v_l, v_r in zip(buffer_shape, arg_shape):
            if isinstance(v_l, tir.Var):
                var_map[v_l] = v_r
            elif not isinstance(v_l, tir.IntImm):
                return [], False

    ret_tensors = call.sinfo_args[0]
    ret_tensors = (
        [ret_tensors]  # type: ignore[assignment]
        if isinstance(ret_tensors, relax.TensorStructInfo)
        else list(ret_tensors.fields)
    )
    for i, ret_tensor in enumerate(ret_tensors):
        buffer_shape = func.buffer_map[func.params[n_arg + i]].shape
        ret_tensor_shape = ret_tensor.shape.values
        assert len(buffer_shape) == len(ret_tensor_shape)
        for v_l, v_r in zip(buffer_shape, ret_tensor_shape):
            if isinstance(v_l, tir.Var):
                var_map[v_l] = v_r
            elif not isinstance(v_l, tir.IntImm):
                return [], False

    updated_tensor_sinfo = []
    for sinfo in tensor_sinfo:
        if not _has_symbolic_var(sinfo):
            updated_tensor_sinfo.append(sinfo)
            continue
        new_shape = []
        for dim in sinfo.shape.values:
            new_shape.append(tir.stmt_functor.substitute(dim, var_map))
        updated_tensor_sinfo.append(relax.TensorStructInfo(new_shape, sinfo.dtype))
    return updated_tensor_sinfo, True
