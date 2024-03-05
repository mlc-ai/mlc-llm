import re

from typing import List, Union

import tvm
from tvm.ir import GlobalVar


def SetEntryFuncs(*entry_funcs: List[Union[GlobalVar, str]]) -> tvm.ir.transform.Pass:
    """Update which functions are externally-exposed

    All functions whose GlobalVar is contained `entry_funcs` list, or
    whose name matches a regular expression in `entry_funcs`, are set
    as externally exposed.  All other functions are set as internal.

    This pass does not add or remove any functions from the
    `IRModule`.  This pass may result in functions no longer being
    used by any externally-exposed function.  In these cases, users
    may use the `relax.transform.DeadCodeElimination` pass to remove
    any unnecessary functions.

    Parameters
    ----------
    entry_funcs: List[Union[GlobalVar, str]]

        Specifies which functions that should be externally exposed,
        either by GlobalVar or by regular expression.

    Returns
    -------
    transform: tvm.ir.transform.Pass

        The IRModule-to-IRModule transformation
    """

    def is_entry_func(gvar: GlobalVar) -> bool:
        for entry_func in entry_funcs:
            if isinstance(entry_func, GlobalVar):
                if entry_func.same_as(gvar):
                    return True
            elif isinstance(entry_func, str):
                if re.fullmatch(entry_func, gvar.name_hint):
                    return True
            else:
                raise TypeError(
                    f"SetEntryFuncs requires all arguments to be a GlobalVar or a str.  "
                    f"However, argument {entry_func} has type {type(entry_func)}."
                )

    def is_exposed(func: tvm.ir.BaseFunc) -> bool:
        return func.attrs is not None and "global_symbol" in func.attrs

    @tvm.ir.transform.module_pass(opt_level=0, name="SetEntryFuncs")
    def transform(mod: tvm.IRModule, _pass_context) -> tvm.IRModule:
        updates = {}
        for gvar, func in mod.functions.items():
            if is_entry_func(gvar):
                if not is_exposed(func):
                    updates[gvar] = func.with_attr("global_symbol", gvar.name_hint)
            else:
                if is_exposed(func):
                    updates[gvar] = func.without_attr("global_symbol")

        if updates:
            mod = mod.clone()
            mod.update(updates)

        return mod

    return transform
