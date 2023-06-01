from typing import Callable, Dict, List, Set, Tuple

import tvm
from tvm import relax
from tvm.ir.module import IRModule

"""
This pass in this file reorders the bindings of the weight transform function
according to the weight location in binary files. The goal of the reorder is to
reduce the memory pressure when loading the raw model weights and processing
them. In the ideal case, with this pass, the highest CPU memory usage will
around the size of the largest raw weight binary file.

Regarding the implementation, the bindings of fetching a raw weight in the
weight transform function are all in the form of `lv = params[idx]`. Here, each
index specifies a raw weight tensor, and the raw weight tensor resides in a
binary file on the disk.

We group such `lv = params[idx]` into multiple groups, such that all raw weight
tensors in a group come from a same binary file. We reorder the bindings
according to the grouping result based on topological sort.

In ideal case, after reordering the weight transform function has the following
process during execution:
* load a weight binary file,
* process all weights in this file,
* load another weight binary file,
* process all weights in this file,
* ...

So the maximum CPU memory usage will be the size of the largest raw weight
binary file, since we process and release all the raw weight tensors immediately
after loading them from the file.
"""


def analyze_func(
    func: relax.Function,
    pidx2binname: Dict[int, str],
) -> Tuple[
    List[relax.Binding],
    Dict[relax.Var, List[relax.Binding]],
    Dict[relax.Binding, int],
]:
    """Binding grouping analysis function.
    It takes the function to be analyzed, and mapping from each raw tensor index
    to the name of the binary file where it resides.

    This analysis function
    * computes a new order of weight fetching bindings (the bindings in form
    `lv = params[idx]`) based on weight location on disk.
    * collects the dataflow def-use information of the given function for
    topological sort (particularly, it collects the consumers of each binding
    variables and the number of variables each binding depends on).

    Parameters
    ----------
    func : relax.Function
        The weight transform function to be analyzed.

    pidx2binname : Dict[int, str]
        The mapping from each raw tensor index to the name of the binary
        file where it resides.

    Returns
    -------
    get_param_bindings : List[relax.Binding]
        The weight fetching bindings (`lv = params[idx]`) in the new order.

    var_users : Dict[relax.Var, List[relax.Binding]]
        The consumer bindings of each binding variable.
        Used for topological sort.

    num_depending_vars : Dict[relax.Binding, int]
        The number of variables each binding depends on.
        Used for topological sort.
    """

    # The mapping of the weight fetching bindings in each binary file.
    # Here empty string means the weight is not in any binary file (e.g., cached
    # sin and cos values for rotary embeddings).
    binname2get_param_bindings: Dict[str, List[relax.Binding]] = {"": []}
    # The set of binding variables.
    binding_var_set: Set[relax.Var] = set()
    var_users: Dict[relax.Var, List[relax.Binding]] = {}
    num_depending_vars: Dict[relax.Binding, int] = {}

    # Sanity check on the function pattern.
    assert len(func.params) == 1
    assert isinstance(func.body, relax.SeqExpr)
    assert len(func.body.blocks) == 1
    assert isinstance(func.body.blocks[0], relax.DataflowBlock)
    assert func.body.blocks[0].bindings[-1].var.same_as(func.body.body)

    params = func.params[0]
    bindings = func.body.blocks[0].bindings

    # Go through each binding except the last one. (The last one is the output
    # binding `gv = (lv, lv1, ...)`) which we ignore for analysis.
    for binding in bindings[:-1]:
        value = binding.value
        binding_var_set.add(binding.var)
        var_users[binding.var] = []

        if isinstance(value, relax.TupleGetItem) and value.tuple_value.same_as(params):
            # For weight fetching bindings (`lv = params[idx]`), we group them
            # according to the binary file name.
            pidx = value.index
            if pidx not in pidx2binname:
                binname2get_param_bindings[""].append(binding)
                continue

            binname = pidx2binname[pidx]
            if binname in binname2get_param_bindings:
                binname2get_param_bindings[binname].append(binding)
            else:
                binname2get_param_bindings[binname] = [binding]
        else:
            # For other bindings, we collect the use-def information for
            # topological sort.
            num_depending_vars[binding] = 0

            def fvisit(obj):
                if isinstance(obj, relax.Var) and obj in binding_var_set:
                    assert obj in var_users
                    var_users[obj].append(binding)
                    num_depending_vars[binding] += 1

            relax.analysis.post_order_visit(value, fvisit)

    # Get the weight fetching bindings in new order according to the group results.
    get_param_bindings: List[relax.Binding] = []
    for bindings in binname2get_param_bindings.values():
        get_param_bindings += bindings

    return get_param_bindings, var_users, num_depending_vars


def reorder_func(
    func: relax.Function,
    pidx2binname: Dict[int, str],
) -> relax.Function:
    """Reorder the bindings of the input weight transform Relax function
    according the weight location in binary files.

    This function first analyzes the input function and gets the reordered
    weight fetching bindings and the use-def information for topological sort.
    It then reorders all bindings in the function with topological sort.

    Parameters
    ----------
    func : relax.Function
        The weight transform function to be analyzed.

    pidx2binname : Dict[int, str]
        The mapping from each raw tensor index to the name of the binary
        file where it resides.

    Returns
    -------
    func_updated : relax.Function
        The returned function where the bindings are updated with the new order.
    """
    get_param_bindings, var_users, num_depending_vars = analyze_func(func, pidx2binname)

    # The bindings in the new order, output by the topological sort.
    new_bindings: List[relax.Binding] = []
    # The queue used in the topological sort.
    binding_queue: List[relax.Binding] = []

    for binding, n_depending in list(num_depending_vars.items()):
        if n_depending == 0:
            binding_queue.append(binding)
            del num_depending_vars[binding]

    # Start topological sort:
    #   each time we emit a weight fetching binding, and then adds all bindings
    #   that depend on it.
    for get_param_binding in get_param_bindings:
        binding_queue.append(get_param_binding)

        while len(binding_queue) > 0:
            binding = binding_queue.pop(0)
            new_bindings.append(binding)
            for user_binding in var_users[binding.var]:
                num_depending_vars[user_binding] -= 1
                if num_depending_vars[user_binding] == 0:
                    del num_depending_vars[user_binding]
                    binding_queue.append(user_binding)

    # Add the output binding.
    new_bindings.append(func.body.blocks[0].bindings[-1])
    # Sanity check on the integrity.
    assert len(new_bindings) == len(func.body.blocks[0].bindings)
    assert len(num_depending_vars) == 0

    return relax.Function(
        func.params,
        relax.SeqExpr(blocks=[relax.DataflowBlock(new_bindings)], body=func.body.body),
        func.ret_struct_info,
        func.is_pure,
        func.attrs,
    )


@tvm.transform.module_pass(opt_level=0, name="ReorderTransformFunc")
class ReorderTransformFunc:
    def __init__(
        self,
        pidx2pname: Dict[int, str],
        pname2binname: Dict[str, str],
        f_convert_pname_fwd: Callable[[str], str],
    ) -> None:
        self.pidx2binname: Dict[int, str] = {
            pidx: pname2binname[f_convert_pname_fwd(pname)]
            for pidx, pname in pidx2pname.items()
        }

    def transform_module(
        self, mod: IRModule, ctx: tvm.transform.PassContext
    ) -> IRModule:
        for gv, func in list(mod.functions.items()):
            if isinstance(func, relax.Function):
                assert gv.name_hint.endswith("_transform_params")
                func_updated = reorder_func(func, self.pidx2binname)
                mod[gv] = func_updated
        return mod
