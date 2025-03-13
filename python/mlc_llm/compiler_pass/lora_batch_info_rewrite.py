import tvm
from tvm import relax


@relax.expr_functor.mutator
class LoraBatchInfoRewriter(relax.PyExprMutator):
    """ Set paged_kv_cache var as param of packed func "vm.builtin.kv_state_get_lora_batch_info"."""

    def __init__(self, mod):
        super().__init__(mod)
        self.mod = mod
        self.paged_kv_cache = None

    def visit_call_(self, call):
        if call.op == tvm.ir.Op.get("relax.call_pure_packed"):
            if call.args[0].global_symbol == "vm.builtin.kv_state_get_lora_batch_info":
                assert self.paged_kv_cache is not None
                return relax.Call(call.op, [call.args[0], self.paged_kv_cache], call.attrs, call.sinfo_args)
        return super().visit_call_(call)

    def transform(self):
        for gv, func in self.mod.functions.items():
            if isinstance(func, relax.Function):
                self.paged_kv_cache = None
                for param in func.params:
                    if param.name_hint == "paged_kv_cache":
                        self.paged_kv_cache = param
                        break
                new_func = self.visit_expr(func)
                self.builder_.update_func(gv, new_func)
        return self.builder_.get()


@tvm.transform.module_pass(opt_level=0, name="LoraBatchInfoRewrite")
class LoraBatchInfoRewrite:
    """ The pass is to set kv_cache var as param of packed func "vm.builtin.kv_state_get_lora_batch_info".
    """
    def transform_module(self, mod, ctx):
        return LoraBatchInfoRewriter(mod).transform()