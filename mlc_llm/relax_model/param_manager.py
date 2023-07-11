from typing import Callable, Dict, List, Set, Tuple

import tvm
from tvm import relax, tir
from tvm.relax.expr import Expr, Function, Var
from tvm.relax.testing import nn
from tvm.relax.analysis import remove_all_unused
from tvm.relax.expr_functor import PyExprMutator, mutator

from .. import quantization
from .modules import named_parameters


class Parameter:
    """The abstraction of weight tensors (e.g., linear layer weight, embedding
    table, etc.) in a model.

    Attributes
    ----------
    name : str
        The name of the parameter.
        The name of a weight is got by `named_parameters()` method, similar to
        PyTorch's `named_parameters()` function.
        An example name is `model.layers.11.self_attn.k_proj.weight`.
        In a model, the name is the **unique** identifier of a parameter.

    param_info : relax.TensorStructInfo
        The shape and dtype of the parameter.
        The shape can be accessed by `param_info.shape`, which is a relax.ShapeExpr instance.
        And the dtype can be accessed by `param_info.dtype`, which is a Python string.

    quant_spec : quantization.QuantizationSpec
        The quantization specification of this parameter.
        It specifies the algorithm to quantize and dequantize this parameter (or
        this parameter does not need quantization).
    """

    name: str
    param_info: relax.TensorStructInfo
    quant_spec: quantization.QuantizationSpec

    def __init__(
        self,
        name: str,
        param_info: relax.TensorStructInfo,
        quant_spec: quantization.QuantizationSpec,
    ) -> None:
        self.name = name
        self.param_info = param_info
        self.quant_spec = quant_spec


class ParamManager:
    """The model-wise data structure which contains the information of every
    weight in the model and is in charge of applying quantization and dequantization
    to the parameters at the entire model level.

    Attributes
    ----------
    params : Dict[str, Parameter]
        The mapping from parameter names to parameters.

    param_names : List[str]
        The name list of all the parameters.
        To enforce a unique order or all the parameters for determinism, the
        parameter names are kept in the list, and the parameter order is
        uniquely determined by the parameter name list.

    func_raw_param_map : Dict[relax.Var, Tuple[str, Parameter]]
        The mapping from each relax.Var that denotes a weight parameter to the
        name of the function the var is in (e.g., "prefill" or "decode"), and
        the Parameter it corresponds to.
        This mapping is used for applying quantization transformation to the
        Relax functions (e.g., the "prefill", "decode", etc.) in the model.

    param2qrange : Dict[Parameter, range]
        The mapping from each parameter to the range of its quantized tensors
        in the list of quantized tensors of all parameters.
        Each parameter is quantized into multiple tensors.
        For example, assume we have parameters `p0`, `p1`, `p2`.
        - `p0` is quantized into `t0_0`, `t0_1`,
        - `p1` is quantized into `t1_0`, and
        - `p2` is quantized into `t2_0`, `t2_1` and `t2_2`.
        Then the list of all quantized tensors is `[t0_0, t0_1, t1_0, t2_0, t2_1, t2_2]`,
        and the dict `param2qrange` is
        `{p0: range(0, 2), p1: range(2, 3), p2: range(3, 6)}`.
    """

    params: Dict[str, Parameter]
    param_names: List[str]

    func_raw_param_map: Dict[relax.Var, Tuple[str, Parameter]]
    param2qrange: Dict[Parameter, range]

    def __init__(self) -> None:
        self.params = {}
        self.param_names = []

        self.func_raw_param_map = {}
        self.param2qrange = {}

    def register_params(
        self,
        model: nn.Module,
        func_name: str,
        quantization_scheme: quantization.QuantizationScheme,
        f_get_param_quant_kind: Callable[
            [str, relax.TensorStructInfo], quantization.ParamQuantKind
        ],
    ) -> None:
        """Register the parameters of the input model (within the context of the
        input function) in the parameter manager.

        Parameters
        ----------
        model : nn.Module
            The input model whose parameters are registered.

        func_name : str
            The name of the function the input model is in.
            For example, the "prefill" function or the "decode" function.

        quantization_scheme : quantization.QuantizationScheme
            The quantization scheme of the input model, which describes how
            to quantize the model.

        f_get_param_quant_kind: Callable[[str, relax.TensorStructInfo], quantization.ParamQuantKind]
            A function which takes the name and StructInfo (effectively shape
            and dtype) of a parameter, and returns which quantization kind this
            parameter uses.
            This is used for applying quantization to the parameters.
        """

        # For each parameter in the input model, get its quantization kind and
        # register the parameter with its name and quantization kind.
        for name, param in named_parameters(model).items():
            quant_kind = f_get_param_quant_kind(name, param.struct_info)
            getattr(quantization_scheme, quant_kind.name)
            self.register_param(
                name, param, getattr(quantization_scheme, quant_kind.name), func_name
            )

    def quantization_transform(
        self, mod: tvm.IRModule
    ) -> Tuple[tvm.IRModule, Dict[int, str]]:
        """The entrance function of all quantization-related transformation,
        including creating the function that computes quantization, and
        updating the input IRModule with quantized data as function input and
        dequantization computation.

        Parameters
        ----------
        mod : tvm.IRModule
            The input IRModule to be applied quantization/dequantization.
            The IRModule contains all the constructed Relax functions
            (e.g., the "prefill"/"decode" functions) and is expected to
            have all of its parameters registered in the ParamManager.

        Returns
        -------
        updated_mod : tvm.IRModule
            The IRModule updated with the quantization function and the
            dequantization computation.

        pidx2pname : Dict[int, str]
            The mapping from each parameter's index in the ParamManager
            to the parameter' name.
            This mapping is used for loading weight tensors from disk and
            applying quantization at runtime.
        """
        # Create the quantization function.
        mod_transform = self.create_quantize_func()

        # For each Relax function in the input IRModule (e.g., "prefill"),
        # we create its input relax.Var of all the quantized data, and
        # store the mapping from function name to the var.
        func2param_var: Dict[str, relax.Var] = {}
        for gv, func in mod.functions.items():
            if not isinstance(func, relax.Function):
                continue
            if func.attrs is None or not "num_input" in func.attrs:
                continue
            func2param_var[gv.name_hint] = relax.Var(
                "params", mod_transform["transform_params"].struct_info.ret
            )

        # Cache mapping to avoid duplicate dequantization.
        dequantized_cache: Dict[relax.Var, relax.Var] = {}

        # Define a var replacement function for applying dequantization.
        def f_replace(var: relax.Var, bb: relax.BlockBuilder) -> relax.Var:
            if var in dequantized_cache:
                return dequantized_cache[var]

            assert var in self.func_raw_param_map
            func_name, param = self.func_raw_param_map[var]
            dequantized = self.dequantize(param, func2param_var[func_name], bb)
            dequantized_cache[var] = dequantized
            return dequantized

        # Create the function mutator for applying dequantization.
        replacer = ParamReplacer(mod, func2param_var, f_replace)
        # Update the input IRModule with dequantization.
        mod = replacer.transform()

        # Merge the quantization IRModule into the updated input IRModule.
        for gv, func in mod_transform.functions.items():
            mod[gv] = func

        # Return the merged IRModule and the mapping from each parameter's
        # index to the parameter' name.
        return mod, {pidx: pname for pidx, pname in enumerate(self.param_names)}

    #################### Below are internally called methods ####################

    def register_param(
        self,
        name: str,
        var: relax.Var,
        quant_spec: quantization.QuantizationSpec,
        func_name: str,
    ) -> None:
        """Register a single parameter in the parameter manager.
        In most cases, this method is not directly used outside this class:
        it is called by `register_params` above.

        Parameters
        ----------
        name : str
            The name of the parameter to register.
            Name serves as the unique identifier of the parameter.

        var : relax.Var
            The parameter relax.Var on the nn.Module side.

        quant_spec : quantization.QuantizationSpec
            The quantization specification of the parameter

        func_name : str
            The name of the function the input var is in.
            For example, the "prefill" function or the "decode" function.
        """

        assert (
            var not in self.func_raw_param_map
        ), "The input var is not supposed to be already registered."
        assert isinstance(
            var.struct_info.shape, relax.ShapeExpr
        ), "The parameter to register is expected to have static shape"
        assert all(
            [
                isinstance(dim_len, tir.IntImm)
                for dim_len in var.struct_info.shape.values
            ]
        ), "The parameter to register is expected to have static shape"

        if name in self.params:
            # When the input name appears in `self.params`, it means the input
            # parameter has been previously registered in some other function.
            # Thus, we check if the dtype, shape and the quantization specification
            # of both sides are consistent.
            param = self.params[name]
            assert (
                param.quant_spec == quant_spec
            ), "One parameter is expected to be quantized by single specification in all functions."
            assert (
                param.param_info.dtype == var.struct_info.dtype
            ), "Dtype mismatch of one parameter in two functions."
            assert (
                param.param_info.ndim == var.struct_info.ndim
            ), "Shape mismatch of one parameter in two functions."
            for len0, len1 in zip(
                param.param_info.shape.values, var.struct_info.shape.values
            ):
                assert (
                    len0.value == len1.value
                ), "Shape mismatch of one parameter in two functions."
        else:
            # Otherwise, the parameter is registered for the first time.
            param = Parameter(name, var.struct_info, quant_spec)
            self.params[name] = param
            self.param_names.append(name)

        # Record the mapping from the input relax.Var to the function name and
        # the parameter in the manager.
        self.func_raw_param_map[var] = (func_name, param)

    def create_quantize_func(self) -> tvm.IRModule:
        """Construct the Relax function which computes quantization.
        This method is called by `quantization_transform` below, and is not
        directly invoked outside the class.

        Returns
        -------
        The created function which computes quantization.
        Precisely, an IRModule which contains the main quantization Relax function
        and a series of TIR functions is returned.
        """
        bb = relax.BlockBuilder()

        # Construct the input of the function.
        # Similar to `self.param2qrange`, we need a list of ranges for each
        # parameter to get its corresponding tensors loaded from disk.
        input_tensor_info: List[relax.TensorStructInfo] = []
        loaded_tensor_ranges: List[range] = []
        for name in self.param_names:
            param = self.params[name]
            loaded_tensor_info = param.quant_spec.get_loaded_tensor_info(
                param.param_info
            )
            loaded_tensor_ranges.append(
                range(
                    len(input_tensor_info),
                    len(input_tensor_info) + len(loaded_tensor_info),
                )
            )
            input_tensor_info += loaded_tensor_info
        raw_param_tuple = relax.Var("params", relax.TupleStructInfo(input_tensor_info))

        with bb.function("transform_params", params=[raw_param_tuple]):
            with bb.dataflow():
                quantized_params: List[relax.Var] = []
                for pidx, name in enumerate(self.param_names):
                    param = self.params[name]
                    param_vars: List[relax.Var] = []
                    # Emit relax.TupleGetItem to get the raw parameters or pre-quantized params.
                    for loaded_tensor_idx in loaded_tensor_ranges[pidx]:
                        param_vars.append(
                            bb.emit(
                                relax.TupleGetItem(raw_param_tuple, loaded_tensor_idx)
                            )
                        )

                    # Get the quantization function of this parameter.
                    f_quantize = param.quant_spec.get_quantize_func(param.param_info)
                    if f_quantize is None:
                        # If the parameter does not have a quantization function, either it
                        # does not need quantization or it is pre-quantized.
                        self.param2qrange[param] = range(
                            len(quantized_params),
                            len(quantized_params) + len(param_vars),
                        )
                        quantized_params += param_vars
                    else:
                        # If the parameter has a quantization function, it is not expected
                        # to be pre-quantized.
                        assert len(param_vars) == 1, (
                            "A parameter with quantization function is not expected "
                            "to be pre-quantized."
                        )

                        # Apply the quantization function.
                        quantized_data = bb.emit_te(
                            f_quantize, param_vars[0], primfunc_name_hint="encode"
                        )

                        if isinstance(
                            quantized_data.struct_info, relax.TupleStructInfo
                        ):
                            n_tensor = len(quantized_data.struct_info.fields)
                            assert n_tensor > 1
                            # Record the range of quantized tensors of this parameter.
                            self.param2qrange[param] = range(
                                len(quantized_params), len(quantized_params) + n_tensor
                            )
                            # Collect the quantized tensors to return.
                            for i in range(n_tensor):
                                quantized_params.append(
                                    bb.emit(relax.TupleGetItem(quantized_data, i))
                                )
                        else:
                            assert isinstance(
                                quantized_data.struct_info, relax.TensorStructInfo
                            )
                            self.param2qrange[param] = range(
                                len(quantized_params), len(quantized_params) + 1
                            )
                            quantized_params.append(quantized_data)

                output = bb.emit_output(relax.Tuple(quantized_params))
            bb.emit_func_output(output)

        # Return the created IRModule.
        return bb.get()

    def dequantize(
        self, param: Parameter, quantized_tuple: relax.Var, bb: relax.BlockBuilder
    ) -> relax.Var:
        """Applying dequantization to the input parameter.
        This method is called by `quantization_transform` below, and is not
        directly invoked outside the class.

        Parameters
        ----------
        param : Parameter
            The parameter whose quantized tensors are to be dequantized.

        quantized_tuple : relax.Var
            The relax.Var of the quantized tensors of all parameters in the model.

        bb : relax.BlockBuilder
            The Relax BlockBuilder used for inserting the dequantization computations.

        Returns
        -------
        The dequantized parameter, in the form of a relax.Var.
        """
        # Get the corresponding Relax vars of the quantized tensors of this parameter.
        qparams: List[relax.Var] = []
        for qparam_idx in self.param2qrange[param]:
            qparams.append(bb.emit(relax.TupleGetItem(quantized_tuple, qparam_idx)))

        # Get the dequantization function of this parameter.
        f_dequantize = param.quant_spec.get_dequantize_func(
            param_info=param.param_info,
            qparam_info=[qparam.struct_info for qparam in qparams],
        )
        if f_dequantize is None:
            # If the parameter does not have a dequantization function, its "quantized
            # data" is expected to have only one element.
            assert len(qparams) == 1, (
                "A parameter without dequantization function is expected not to have "
                'more than one "quantized data".'
            )
            return qparams[0]
        else:
            # Apply the dequantization function.
            return bb.emit_te(f_dequantize, *qparams, primfunc_name_hint="decode")


@mutator
class ParamReplacer(PyExprMutator):
    """The function mutator that updates the model with dequantization.

    Attributes
    ----------
    mod : tvm.IRModule
        The IRModule of the model to be updated.

    func2param_var : Dict[str, relax.Var]
        The mapping from each function name to its input var of quantized data tuple.

    f_replace : Callable[[relax.Var, relax.BlockBuilder], relax.Var]
        The function for updating a previous parameter in functions with dequantization.

    param_set : Set[relax.Var]
        The set of previous parameters (before applying quantization and dequantization)
        in the relax functions.
    """

    mod: tvm.IRModule
    func2param_var: Dict[str, relax.Var]
    f_replace: Callable[[relax.Var, relax.BlockBuilder], relax.Var]
    param_set: Set[relax.Var]

    def __init__(
        self,
        mod: tvm.IRModule,
        func2param_var: Dict[str, relax.Var],
        f_replace: Callable[[relax.Var, relax.BlockBuilder], relax.Var],
    ):
        super().__init__(mod)
        self.mod = mod
        self.func2param_var = func2param_var
        self.f_replace = f_replace

    def transform(self) -> tvm.IRModule:
        for gv, func in self.mod.functions.items():
            if not isinstance(func, relax.Function):
                continue
            if func.attrs is None or not "num_input" in func.attrs:
                continue

            assert gv.name_hint in self.func2param_var
            updated_func = self.rewrite_func(func, self.func2param_var[gv.name_hint])
            updated_func = remove_all_unused(updated_func)
            self.builder_.update_func(gv, updated_func)
        return self.builder_.get()

    def rewrite_func(self, func: Function, param_var: relax.Var) -> relax.Function:
        num_input = int(func.attrs["num_input"])
        self.param_set = set(func.params[num_input:])

        body = self.visit_expr(func.body)
        return relax.Function(
            params=func.params[:num_input] + [param_var],
            body=body,
            ret_struct_info=func.ret_struct_info,
            is_pure=func.is_pure,
            attrs=func.attrs,
        ).without_attr("num_input")

    def visit_var_(self, var: Var) -> Expr:
        if var not in self.param_set:
            return super().visit_var_(var)
        return self.f_replace(var, self.builder_)
