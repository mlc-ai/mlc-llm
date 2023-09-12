import json
import os
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import tvm
from torch import Tensor as torchTensor
from tvm import relax, tir
from tvm._ffi.runtime_ctypes import Device
from tvm.relax.analysis import remove_all_unused
from tvm.relax.expr import Expr, Function, Var
from tvm.relax.expr_functor import PyExprMutator, mutator
from tvm.relax.testing import nn

from .. import quantization
from .modules import named_parameters


def f_default_compute_relax_param(relax_pname: str, torch_params: List[Any]) -> Any:
    """The defualt `f_compute_relax_param` for ParamManager.
    See ParamManager for more details.
    """
    raise NotImplementedError()


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

    param_info_dict : Dict[str, relax.TensorStructInfo]
        The shape and dtype of the parameter in each function.
        The shape can be accessed by `param_info_dict[func_name].shape`, which is
        a relax.ShapeExpr instance.
        And the dtype can be accessed by `param_info_dict[func_name].dtype`,
        which is a Python string.

    quant_spec : quantization.QuantizationSpec
        The quantization specification of this parameter.
        It specifies the algorithm to quantize and dequantize this parameter (or
        this parameter does not need quantization).

    shard_dim : Optional[int]
        The dimension to be sharded.
    """

    name: str
    param_info_dict: Dict[str, relax.TensorStructInfo]
    quant_spec: quantization.QuantizationSpec
    shard_dim: Optional[int]

    def __init__(
        self,
        name: str,
        quant_spec: quantization.QuantizationSpec,
        shard_dim: Optional[int],
    ) -> None:
        self.name = name
        self.param_info_dict = dict()
        self.quant_spec = quant_spec
        self.shard_dim = shard_dim

    def register_func(self, func_name: str, param_info: relax.TensorStructInfo):
        self.param_info_dict[func_name] = param_info

    @property
    def param_info(self):
        """Return the shape and dtype of the parameter (in some arbitrary function)."""
        return next(iter(self.param_info_dict.values()))


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

    f_convert_pname_fwd : Callable[[str], List[str]]
        The function which converts Relax parameter name (ours) to torch's
        parameter names, suggesting "to load this Relax parameter, which torch
        parameter(s) are needed".
        - Usually, the function maps a name to itself. For example, in LLaMA we
        map `lm_head.weight` itself, as the parameter has the same name on both
        Relax side and torch side.
        - In some cases we map a name to multiple names. For example, if we
        support combined QKV computing when the torch side separates them, on
        Relax side we only have one QKV weight, while on torch side we have
        one weight for each of Q, K, V. In this case, we map one name to three
        names.
        - In some cases we map a name to a single name which is other than
        itself. This can happen either when the Relax nn.Module has different
        param names than the torch's implementation so we need to map names
        for connection, or when a Relax parameter is computed out from a torch
        parameter. For example, if the torch implementation supports combined
        QKV while the Relax one does not, we need compute the relax parameter
        out from torch's parameter. In this case we map the relax parameter
        name to the torch's parameter name.

    f_convert_param_bkwd : Callable[[str, Any], Optional[List[Tuple[str, Any]]]]
        The function which converts torch parameter and param name back to
        Relax parameters with names. `Any` here stands for numpy.ndarray.
        - Usually, the function just returns the input torch parameter and
        the corresponding Relax parameter's name.
        - In some cases, we return multiple Relax parameters. For example, if
        the torch implementation supports combined QKV while the Relax one does
        not, the function takes torch's combined QKV weight, and return the
        separated Q K V weights with their corresponding names.
        - In some cases we return `None`. This happens when the input torch
        parameter itself does not determine any Relax parameter. For example,
        if we support combined QKV computing when the torch side separates them,
        we return `None` here for the single Q, K, V weights, as by only having
        a Q (or K, V) weight we cannot compute the combined QKV weight.

    f_compute_relax_param : Callable[[str, List[Any]], Any]
        The function which computes a Relax parameter from a list of torch
        parameters. `Any` here stands for numpy.ndarray. In the case when one
        Relax parameter is computed from multiple torch parameters, this
        functions is used.
        For example, if we support combined QKV computing when the torch side
        separates them, we use this function to combine the torch's Q, K, V
        weights into one
        In usual case, this function is not needed and by default it is
        implemented by raising `NotImplementedError` (see f_default_compute_relax_param).

    model_path : str
        The path of the Hugging Face model on disk.

    use_safetensors: bool
        Whether to use `.safetensors` instead of `.bin` to load model.

    safetensors_load_func: Callable[[Union[str, os.PathLike], str], Dict[str, torch.Tensor]]
        A reference to the function `load_file` improted from `safetensors.torch`.
        The goal is to prevent repeatedly importing in a tvm registered function.

    pidx2pname : Dict[int, str]
        The dictionary from each Relax parameter's index in `param_names` to
        the Relax parameter's name.

    torch_pname2binname : Dict[str, str]
        The dictionary from each torch parameter's name to the name of the
        binary shard where the torch parameter is saved.
    """

    params: Dict[str, Parameter]
    param_names: List[str]
    func_raw_param_map: Dict[relax.Var, Tuple[str, Parameter]]
    param2qrange: Dict[Parameter, range]

    qspec_updater_classes: List[quantization.QuantSpecUpdater]

    nparam_to_load: int
    f_convert_pname_fwd: Callable[[str], List[str]]
    f_convert_param_bkwd: Callable[[str, Any], Optional[List[Tuple[str, Any]]]]
    f_compute_relax_param: Callable[[str, List[Any]], Any]
    f_run_prequantize: Optional[Callable[[str], str]]

    model_path: str
    use_safetensors: bool
    safetensors_load_func: Callable[[Union[str, os.PathLike], str], Dict[str, torchTensor]]
    pidx2pname: Dict[int, str]
    torch_pname2binname: Dict[str, str]

    def __init__(self) -> None:
        self.params = {}
        self.param_names = []
        self.params_in_func = {}

        self.func_raw_param_map = {}
        self.param2qrange = None

        self.nparam_to_load = None
        self.f_convert_pname_fwd = None
        self.f_convert_param_bkwd = None
        self.f_compute_relax_param = None
        self.f_run_prequantize = None

        self.qspec_updater_classes = []

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
        if quantization_scheme.qspec_updater_class is not None:
            self.qspec_updater_classes.append(quantization_scheme.qspec_updater_class)
        if quantization_scheme.f_convert_param_bkwd is not None:
            self.f_convert_param_bkwd = quantization_scheme.f_convert_param_bkwd
        if quantization_scheme.f_compute_relax_param is not None:
            self.f_compute_relax_param = quantization_scheme.f_compute_relax_param
        if quantization_scheme.f_run_prequantize is not None:
            self.f_run_prequantize = quantization_scheme.f_run_prequantize

        self.params_in_func[func_name] = []
        # For each parameter in the input model, get its quantization kind and
        # register the parameter with its name and quantization kind.
        for name, relax_param in named_parameters(model).items():
            quant_kind = f_get_param_quant_kind(name, relax_param.struct_info)
            param = self._register_param(
                name,
                relax_param,
                getattr(quantization_scheme, quant_kind.name),
                func_name,
                getattr(relax_param, "shard_dim", None),
            )

            self.params_in_func[func_name].append(param)

    def set_param_loading_func(
        self,
        model_path: str,
        use_safetensors: bool,
        f_convert_pname_fwd: Callable[[str], List[str]] = lambda pname: [pname],
        f_convert_param_bkwd: Callable[
            [str, Any], Optional[List[Tuple[str, Any]]]
        ] = lambda pname, torch_param: [(pname, torch_param)],
        f_compute_relax_param: Callable[[str, List[Any]], Any] = f_default_compute_relax_param,
        *,
        no_lazy_param_loading: bool = False,
    ) -> None:
        """Set the parameter loading functions.

        Parameters
        ----------
        model_path : str
            The path of the Hugging Face model on disk.

        use_safetensors : bool
            Whether to use ``.safetensors`` instead of ``.bin`` to load model.

        f_convert_pname_fwd : Callable[[str], List[str]]
            The function which converts Relax parameter name (ours) to torch's
            parameter names. See the document of ParamManager for more details.

        f_convert_param_bkwd : Callable[[str, Any], Optional[List[Tuple[str, Any]]]]
            The function which converts torch parameter and param name back to
            Relax parameters with names. `Any` here stands for numpy.ndarray.
            See the document of ParamManager for more details.

        f_compute_relax_param : Callable[[str, List[Any]], Any]
            The function which computes a Relax parameter from a list of torch
            parameters. `Any` here stands for numpy.ndarray.
            See the document of ParamManager for more details.

        no_lazy_param_loading : bool
            A boolean indicating that no lazy parameter loading from torch is needed.
            This needs to be set as True when all the model weights are loaded
            at the time of constructing the model.
        """
        self.f_convert_pname_fwd = f_convert_pname_fwd
        if self.f_convert_param_bkwd is None:
            self.f_convert_param_bkwd = f_convert_param_bkwd
        if self.f_compute_relax_param is None:
            self.f_compute_relax_param = f_compute_relax_param

        self.model_path = model_path
        self.use_safetensors = use_safetensors
        if self.use_safetensors:
            # Use a pointer here to prevent repeated import in tvm registered function
            from safetensors.torch import (
                load_file,  # pylint: disable=import-outside-toplevel
            )

            self.safetensors_load_func = load_file

        pnames_to_load = []
        for param_name in self.param_names:
            param = self.params[param_name]
            loaded_names, _ = param.quant_spec.get_loaded_tensor_info(param_name, param.param_info)
            pnames_to_load += loaded_names

        self.nparam_to_load = len(pnames_to_load)
        if not no_lazy_param_loading:
            self.pidx2pname = {pidx: pname for pidx, pname in enumerate(pnames_to_load)}
        else:
            self.pidx2pname = dict()

    def transform_dequantize(self, mod: tvm.IRModule) -> tvm.IRModule:
        """Apply dequantization to the input IRModule.

        Parameters
        ----------
        mod : tvm.IRModule
            The input IRModule to be applied dequantization.
            The IRModule contains all the constructed Relax functions
            (e.g., the "prefill"/"decode" functions) and is expected to
            have all of its parameters registered in the ParamManager.

        Returns
        -------
        updated_mod : tvm.IRModule
            The IRModule updated with the dequantization computation.
        """

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
                "params", self.get_quantized_param_info(gv.name_hint)
            )

        # Cache mapping to avoid duplicate dequantization.
        dequantized_cache: Dict[relax.Var, relax.Var] = {}

        # Define a var replacement function for applying dequantization.
        def f_replace(var: relax.Var, bb: relax.BlockBuilder, func_name: str) -> relax.Var:
            if var in dequantized_cache:
                return dequantized_cache[var]
            assert var in self.func_raw_param_map
            func_name, param = self.func_raw_param_map[var]
            dequantized = self._dequantize(param, func2param_var[func_name], bb, func_name)
            dequantized_cache[var] = dequantized
            return dequantized

        # Create the function mutator for applying dequantization.
        replacer = ParamReplacer(mod, func2param_var, f_replace)
        # Update the input IRModule with dequantization.
        mod = replacer.transform()

        return mod

    def get_quantized_param_info(self, func_name: str) -> List[relax.TensorStructInfo]:
        bb = relax.BlockBuilder()

        self.param2qrange = dict()
        quantized_param_info: List[relax.TensorStructInfo] = []
        for name in self.param_names:
            param = self.params[name]
            param_info = None
            if func_name in param.param_info_dict:
                param_info = param.param_info_dict[func_name]
            else:
                param_info = relax.TensorStructInfo(
                    tvm.ir.load_json(tvm.ir.save_json(param.param_info.shape)),
                    param.param_info.dtype,
                )

            _, loaded_tensor_info = param.quant_spec.get_loaded_tensor_info(name, param_info)

            provided_tensor_vars: List[relax.Var] = []
            for provided_info in loaded_tensor_info:
                provided_tensor_vars.append(relax.Var("var", provided_info))

            # Get the quantization function of this parameter.
            f_quantize = param.quant_spec.get_quantize_func(param_info)
            if f_quantize is None:
                # If the parameter does not have a quantization function, either it
                # does not need quantization or it is pre-quantized.
                self.param2qrange[param] = range(
                    len(quantized_param_info),
                    len(quantized_param_info) + len(provided_tensor_vars),
                )
                quantized_param_info += [var.struct_info for var in provided_tensor_vars]
            else:
                # If the parameter has a quantization function, it is not expected
                # to be pre-quantized.
                assert len(provided_tensor_vars) == 1, (
                    "A parameter with quantization function is not expected " "to be pre-quantized."
                )

                # Apply the quantization function.
                quantized_data = bb.normalize(f_quantize(bb, provided_tensor_vars))
                if isinstance(quantized_data.struct_info, relax.TupleStructInfo):
                    n_tensor = len(quantized_data.struct_info.fields)
                    assert n_tensor > 1
                    # Record the range of quantized tensors of this parameter.
                    self.param2qrange[param] = range(
                        len(quantized_param_info),
                        len(quantized_param_info) + n_tensor,
                    )
                    # Collect the quantized tensors to return.
                    for i in range(n_tensor):
                        quantized_param_info.append(
                            relax.TupleGetItem(quantized_data, i).struct_info
                        )
                else:
                    assert isinstance(quantized_data.struct_info, relax.TensorStructInfo)
                    self.param2qrange[param] = range(
                        len(quantized_param_info), len(quantized_param_info) + 1
                    )
                    quantized_param_info.append(quantized_data.struct_info)

        return relax.TupleStructInfo(quantized_param_info)

    def get_param_loading_functions(
        self,
        model_params: List[Optional[tvm.nd.NDArray]],
        loaded_params: List[tvm.nd.NDArray],
        loaded_idx_set: Set[int],
        loaded_torch_bins: Set[str],
        cached_relax_params: Dict[int, tvm.nd.NDArray],
        cached_torch_params: Dict[str, Any],
        device: Device,
        device_cpu: Device,
    ) -> Tuple[Callable, Callable]:
        """A wrapper function which returns the `get_item` and `set_item`
        functions for parameter lazy loading.

        Parameters
        ----------
        model_params : List[Optional[tvm.nd.NDArray]]
            The pre-loaded model parameters, for which we skip lazy loading.

        loaded_params : List[tvm.nd.NDArray]
            The parameter loading result, storing all the loaded parameters.

        loaded_idx_set : Set[int]
            The set of indices of loaded parameters, serving for robustness
            guarantee to avoid one parameter being loaded for multiple times.

        loaded_torch_bins : Set[str]
            The set of torch binary filenames, serving for robustness guarantee
            to avoid one torch binary file being loaded for multiple times.

        cached_relax_params : Dict[int, tvm.nd.NDArray]
            The set of cached Relax parameters.

        cached_torch_params: Dict[str, Any]
            The set of cached torch parameters. `Any` here stands for numpy.ndarray.

        device : Device
            The device which we load the parameters to.

        device_cpu : Device
            The CPU device.
        """
        import torch  # pylint: disable=import-outside-toplevel

        assert self.f_convert_pname_fwd is not None
        assert self.f_convert_param_bkwd is not None
        assert self.f_compute_relax_param is not None
        pname2pidx: Dict[str, int] = {pname: pidx for pidx, pname in self.pidx2pname.items()}

        def fetch_torch_param(torch_param):
            if str(torch_param.dtype) == "torch.bfloat16":
                # Convert to float32 first.
                return torch_param.detach().cpu().float().numpy()
            else:
                return torch_param.detach().cpu().numpy()

        def load_torch_params_from_bin(torch_binname: str):
            torch_binpath = os.path.join(self.model_path, torch_binname)
            torch_params = None
            if self.use_safetensors:
                torch_params = self.safetensors_load_func(torch_binpath)
            else:
                torch_params = torch.load(
                    torch_binpath,
                    map_location=torch.device("cpu"),
                )
            torch_param_names = list(torch_params.keys())
            for torch_param_name in torch_param_names:
                torch_param = fetch_torch_param(torch_params[torch_param_name])
                del torch_params[torch_param_name]

                relax_params = self.f_convert_param_bkwd(torch_param_name, torch_param)
                if relax_params is not None:
                    for param_name, param in relax_params:
                        if param_name not in pname2pidx.keys():
                            continue
                        pidx = pname2pidx[param_name]
                        assert pidx not in cached_relax_params
                        cached_relax_params[pidx] = tvm.nd.array(param, device_cpu)
                else:
                    assert torch_param_name not in cached_torch_params
                    cached_torch_params[torch_param_name] = torch_param
                del torch_param

        def get_item(i):
            # If the weight is already provided by `model_params`, directly use it
            # and no need to load from binary file.
            if model_params[i] is not None:
                assert i not in cached_relax_params
                return tvm.nd.array(model_params[i], device=device)

            # Otherwise, we load the weight from its corresponding binary file.
            assert i in self.pidx2pname
            relax_pname = self.pidx2pname[i]
            torch_pnames = self.f_convert_pname_fwd(relax_pname)

            if i not in cached_relax_params:
                for torch_binname in [
                    self.torch_pname2binname[torch_pname] for torch_pname in torch_pnames
                ]:
                    if torch_binname in loaded_torch_bins:
                        continue
                    load_torch_params_from_bin(torch_binname)
                    loaded_torch_bins.add(torch_binname)

            if i not in cached_relax_params:
                assert len(torch_pnames) > 1
                assert all([torch_pname in cached_torch_params] for torch_pname in torch_pnames)
                cached_relax_params[i] = self.f_compute_relax_param(
                    relax_pname,
                    [cached_torch_params[torch_pname] for torch_pname in torch_pnames],
                )
                for torch_pname in torch_pnames:
                    del cached_torch_params[torch_pname]

            assert i in cached_relax_params
            assert i not in loaded_idx_set
            param_on_device = tvm.nd.array(cached_relax_params[i], device=device)
            loaded_idx_set.add(i)
            del cached_relax_params[i]
            return param_on_device

        def set_item(i, computed_param):
            if len(loaded_params) <= i:
                loaded_params.extend([None for _ in range(i - len(loaded_params) + 1)])
            loaded_params[i] = tvm.nd.array(computed_param, device=device_cpu)

        return get_item, set_item

    #################### Below are internally called methods ####################

    def _register_param(
        self,
        name: str,
        var: relax.Var,
        quant_spec: quantization.QuantizationSpec,
        func_name: str,
        shard_dim: Optional[int],
    ) -> Parameter:
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

        shard_dim : int
            The dimension along which the parameter is sharded.

        Returns
        -------
        param : Parameter
            The registered Parameter.
        """
        assert (
            var not in self.func_raw_param_map
        ), "The input var is not supposed to be already registered."
        assert isinstance(
            var.struct_info.shape, relax.ShapeExpr
        ), "The parameter to register is expected to have shape as a tuple"

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
            for len0, len1 in zip(param.param_info.shape.values, var.struct_info.shape.values):
                if isinstance(len0, tir.IntImm) and isinstance(len1, tir.IntImm):
                    assert (
                        len0.value == len1.value
                    ), "Shape mismatch of one parameter in two functions."
        else:
            # Otherwise, the parameter is registered for the first time.
            param = Parameter(name, quant_spec, shard_dim)
            self.params[name] = param
            self.param_names.append(name)

        param.register_func(func_name, var.struct_info)
        # Record the mapping from the input relax.Var to the function name and
        # the parameter in the manager.
        self.func_raw_param_map[var] = (func_name, param)
        return param

    def _dequantize(
        self,
        param: Parameter,
        quantized_tuple: relax.Var,
        bb: relax.BlockBuilder,
        func_name: str,
        qparams: List[relax.Var] = None,
    ) -> relax.Var:
        """Applying dequantization to the input parameter.
        This method is called by `transform_module` below, and is not
        directly invoked outside the class.

        Parameters
        ----------
        param : Parameter
            The parameter whose quantized tensors are to be dequantized.

        quantized_tuple : relax.Var
            The relax.Var of the quantized tensors of all parameters in the model.

        bb : relax.BlockBuilder
            The Relax BlockBuilder used for inserting the dequantization computations.

        func_name : str
            The name of the  function which dequantization is applied to.

        qparams : List[relax.Var]
            The quantized parts of the parameter.
            By default it is `None`, in which case we will get the quantized parts
            from `quantized_tuple`.

        Returns
        -------
        The dequantized parameter, in the form of a relax.Var.
        """
        if not qparams:
            # Get the corresponding Relax vars of the quantized tensors of this parameter.
            qparams: List[relax.Var] = []
            for qparam_idx in self.param2qrange[param]:
                qparams.append(bb.emit(relax.TupleGetItem(quantized_tuple, qparam_idx)))

        # Get the dequantization function of this parameter.
        f_dequantize = param.quant_spec.get_dequantize_func(
            param_info=param.param_info_dict[func_name],
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
            return bb.emit(f_dequantize(bb, qparams))


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

    cur_func_name: str

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
        self.cur_func_name = ""

    def transform(self) -> tvm.IRModule:
        for gv, func in self.mod.functions.items():
            if not isinstance(func, relax.Function):
                continue
            if func.attrs is None or not "num_input" in func.attrs:
                continue

            assert (
                gv.name_hint in self.func2param_var
            ), f"{gv.name_hint} not in {self.func2param_var}"
            self.cur_func_name = gv.name_hint
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
        return self.f_replace(var, self.builder_, self.cur_func_name)


##################################################################


def load_torch_pname2binname_map(
    model_path: str,
    use_safetensors: bool,
    relax_pnames: Set[str],
    f_convert_pname_fwd: Callable[[str], List[str]] = lambda pname: [pname],
) -> Dict[str, str]:
    """Constructing the dictionary from each torch parameter's name to
    the name of the binary shard where the torch parameter is saved.

    Parameters
    ----------
    model_path : str
        The path of the Hugging Face model on disk.

    use_safetensors: bool
        Whether to use ``.safetensors`` instead of ``.bin`` to load model.

    relax_pnames: Set[str]
        The name of the Relax parameters.

    f_convert_pname_fwd: Callable[[str], List[str]]
        The function which converts Relax parameter name to torch's
        parameter names. See ParamManager for more details.
    """
    bin_idx_path = None
    single_shard_file_name = None
    if use_safetensors:
        bin_idx_path = os.path.join(model_path, "model.safetensors.index.json")
        single_shard_file_name = "model.safetensors"
    else:
        bin_idx_path = os.path.join(model_path, "pytorch_model.bin.index.json")
        single_shard_file_name = "pytorch_model.bin"
    single_shard_path = os.path.join(model_path, single_shard_file_name)

    if os.path.isfile(bin_idx_path):
        # Multiple weight shards.
        with open(bin_idx_path, "r") as f_torch_json:
            torch_bin_json = json.load(f_torch_json)
            torch_pname2binname = torch_bin_json["weight_map"]
    elif os.path.isfile(single_shard_path):
        # Single weight shard.
        torch_pname2binname = {
            torch_pname: single_shard_file_name
            for relax_pname in relax_pnames
            for torch_pname in f_convert_pname_fwd(relax_pname)
        }
    else:
        suffix = ".safetensors" if use_safetensors else ".bin"
        shard_names = []
        # Collect Scan every single file with the suffix
        for filename in os.listdir(model_path):
            if filename.endswith(suffix):
                shard_names.append(filename)
        if len(shard_names) == 1:
            torch_pname2binname = {
                torch_pname: shard_names[0]
                for relax_pname in relax_pnames
                for torch_pname in f_convert_pname_fwd(relax_pname)
            }
        else:
            raise ValueError("Multiple weight shard files without json map is not supported")
    return torch_pname2binname


def create_quantize_func(param_manager: ParamManager) -> tvm.IRModule:
    """Construct the Relax function which computes quantization.
    This method is called by `transform_module` below, and is not
    directly invoked outside the class.

    Parameters
    ----------
    param_manager : ParamManager
        The parameter manager which has all the parameter information.

    Returns
    -------
    The created function which computes quantization.
    Precisely, an IRModule which contains the main quantization Relax function
    and a series of TIR functions is returned.
    """
    bb = relax.BlockBuilder()
    param2qrange = dict()

    # Construct the input of the function.
    # We need a list of ranges for each
    # parameter to get its corresponding tensors loaded from disk.
    input_tensor_info: List[relax.TensorStructInfo] = []
    loaded_tensor_ranges: List[range] = []
    for name in param_manager.param_names:
        param = param_manager.params[name]
        _, loaded_tensor_info = param.quant_spec.get_loaded_tensor_info(name, param.param_info)
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
            for pidx, name in enumerate(param_manager.param_names):
                param = param_manager.params[name]
                param_vars: List[relax.Var] = []
                # Emit relax.TupleGetItem to get the raw parameters or pre-quantized params.
                for loaded_tensor_idx in loaded_tensor_ranges[pidx]:
                    param_vars.append(
                        bb.emit(relax.TupleGetItem(raw_param_tuple, loaded_tensor_idx))
                    )

                # Get the quantization function of this parameter.
                f_quantize = param.quant_spec.get_quantize_func(param.param_info)
                if f_quantize is None:
                    # If the parameter does not have a quantization function, either it
                    # does not need quantization or it is pre-quantized.
                    param2qrange[param] = range(
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
                    quantized_data = bb.emit(f_quantize(bb, param_vars))

                    if isinstance(quantized_data.struct_info, relax.TupleStructInfo):
                        n_tensor = len(quantized_data.struct_info.fields)
                        assert n_tensor > 1
                        # Record the range of quantized tensors of this parameter.
                        param2qrange[param] = range(
                            len(quantized_params), len(quantized_params) + n_tensor
                        )
                        # Collect the quantized tensors to return.
                        for i in range(n_tensor):
                            quantized_params.append(bb.emit(relax.TupleGetItem(quantized_data, i)))
                    else:
                        assert isinstance(quantized_data.struct_info, relax.TensorStructInfo)
                        param2qrange[param] = range(
                            len(quantized_params), len(quantized_params) + 1
                        )
                        quantized_params.append(quantized_data)

            output = bb.emit_output(relax.Tuple(quantized_params))
        bb.emit_func_output(output)

    mod = bb.get()
    param_manager.param2qrange = param2qrange
    # Return the created IRModule.
    return bb.get()
