from typing import Callable, Dict, List, Set, Tuple, Optional

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
) -> Tuple[List[relax.Binding], Dict[relax.Var, List[relax.Binding]], Dict[relax.Binding, int],]:
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

    if func.attrs is not None and "num_input" in func.attrs:
        num_input = func.attrs["num_input"].value
    else:
        num_input = 0

    # Sanity check on the function pattern.
    assert isinstance(func.body, relax.SeqExpr)
    assert len(func.body.blocks) == 1
    assert isinstance(func.body.blocks[0], relax.DataflowBlock)
    assert func.body.blocks[0].bindings[-1].var.same_as(func.body.body)

    if isinstance(func.params[num_input].struct_info, relax.TupleStructInfo):
        model_param_tuple = func.params[num_input]
    else:
        model_param_tuple = None
        for i, var in enumerate(func.params[num_input:]):
            binname = pidx2binname.get(i, var.name_hint)
            if binname not in binname2get_param_bindings:
                binname2get_param_bindings[binname] = []
            binname2get_param_bindings[binname].append(var)

    bindings = list(func.body.blocks[0].bindings)

    # Go through each binding except the last one. (The last one is the output
    # binding `gv = (lv, lv1, ...)`) which we ignore for analysis.
    for binding in bindings[:-1]:
        value = binding.value
        binding_var_set.add(binding.var)
        var_users[binding.var] = []

        if (
            model_param_tuple is not None
            and isinstance(value, relax.TupleGetItem)
            and value.tuple_value.same_as(model_param_tuple)
        ):
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
    pidx2binname: Optional[Dict[int, str]] = None,
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

    pidx2binname : Optional[Dict[int, str]]

        The mapping from each raw tensor index to the name of the
        binary file where it resides.  If a relax dataflow graph has
        multiple valid topological sorts, the order that minimizes the
        number of simultaneously open files will be produced

        If `None` (default), the existing order of relax bindings is
        preserved in these cases.

    Returns
    -------
    func_updated : relax.Function
        The returned function where the bindings are updated with the new order.

    """

    if pidx2binname is None:
        pidx2binname = {}

    bindings_to_visit = list(func.body.blocks[0].bindings)
    param_lookup = {param: i for i, param in enumerate(func.params)}
    binding_lookup = {}
    previously_defined = set(func.params)
    new_binding_order = []

    param_tuple = None
    if len(func.params) == 1 and isinstance(func.params[0].struct_info, relax.TupleStructInfo):
        param_tuple = func.params[0]

    def sort_key(i):
        binding = bindings_to_visit[i]
        upstream_vars = relax.analysis.free_vars(binding.value)

        valid_ordering = all(var in previously_defined for var in upstream_vars)
        last_param_used = max(
            (param_lookup[var] for var in upstream_vars if var in param_lookup), default=-1
        )
        earliest_binding_used = min(
            (binding_lookup[var] for var in upstream_vars if var in binding_lookup), default=-1
        )
        if (
            param_tuple
            and isinstance(binding.value, relax.TupleGetItem)
            and binding.value.tuple_value.same_as(param_tuple)
            and binding.value.index in pidx2binname
        ):
            tuple_param_group = pidx2binname[binding.value.index]
        else:
            tuple_param_group = ""

        return [
            # First, sort by valid orderings, so the min element will
            # always be a binding that would be legal to use.
            -valid_ordering,
            # Next, sort by the function parameter used by this
            # binding, in increasing order.  That way, we start by
            # computing everything that required just the first
            # parameter, then move on to variables that can be
            # computed with the first two parameters, and so on.
            last_param_used,
            # Next, sort by the other bindings used.  This way, for
            # variables that are only used as input in a single
            # downstream binding, the variable's required live range
            # is minimized.
            -earliest_binding_used,
            # Finally, if this is a `TupleGetItem(param_tuple, i)`,
            # select the option that uses an already-open file.  This
            # is mainly used relevant when loading from pytorch, which
            # require loading the entire file at once.
            tuple_param_group,
        ]

    while bindings_to_visit:
        i_binding = min(range(len(bindings_to_visit)), key=sort_key)
        binding = bindings_to_visit.pop(i_binding)

        assert all(var in previously_defined for var in relax.analysis.free_vars(binding.value))
        new_binding_order.append(binding)
        previously_defined.add(binding.var)

    assert len(new_binding_order) == len(func.body.blocks[0].bindings)

    return relax.Function(
        func.params,
        relax.SeqExpr(
            blocks=[relax.DataflowBlock(new_binding_order)],
            body=func.body.body,
        ),
        func.ret_struct_info,
        func.is_pure,
        func.attrs,
    )


@tvm.transform.module_pass(opt_level=0, name="ReorderTransformFunc")
class ReorderTransformFunc:
    def __init__(self, pidx2binname: Optional[Dict[int, str]] = None):
        if pidx2binname is None:
            pidx2binname = {}
        self.pidx2binname = pidx2binname

    def transform_module(
        self,
        mod: IRModule,
        ctx: tvm.transform.PassContext,
    ) -> IRModule:
        mod = mod.clone()
        for gv, func in list(mod.functions.items()):
            if isinstance(func, relax.Function) and func.attrs and "global_symbol" in func.attrs:
                assert gv.name_hint.endswith("transform_params")
                func_updated = reorder_func(func, self.pidx2binname)
                mod[gv] = func_updated
        return mod
