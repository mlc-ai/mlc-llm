import os
import argparse
import numpy as np
from tqdm import tqdm
from enum import Enum
from transformers import AutoTokenizer
from typing import List, Dict, Any, Union, Callable
from datasets import load_dataset

import tvm
from tvm import relax
from tvm import dlight as dl

import mlc_llm
from mlc_llm import utils
from mlc_llm.utils import load_params
from ..transform.smoothquant import SCALE_PREFIX_NAME, ZP_PREFIX_NAME


# List of supported calibration datasets.
dataset_list = ["dummy", "piqa", "gsm8k"]


def get_runtime_func(funcs: List[str], mod: tvm.IRModule):
    target = tvm.target.Target.current(allow_none=False)
    target_kind = target.kind.default_keys[0]
    vm_device = tvm.device(target_kind)

    # This code is used to speedup calibration process.
    lowered_mod = tvm.relax.pipeline.get_pipeline()(mod)
    lowered_mod = tvm.relax.transform.DeadCodeElimination(funcs)(lowered_mod)
    if target_kind != "cpu":
        with target:
            lowered_mod = dl.ApplyDefaultSchedule(
                dl.gpu.Matmul(),
                dl.gpu.GEMV(),
                dl.gpu.Reduction(),
                dl.gpu.GeneralReduction(),
                dl.gpu.Fallback(),
            )(lowered_mod)

    exe = relax.build(lowered_mod, target)
    vm = relax.VirtualMachine(exe, vm_device)
    runtime_funcs = [vm[fname] for fname in funcs]
    return runtime_funcs[0] if len(funcs) == 1 else runtime_funcs


def to_device(params, device):
    return [param.copyto(device) for param in params]


def _accumulate_outlier_stat(stat, data, func: Callable = np.maximum):
    if stat is None:
        stat = [element.numpy() for element in data]
    else:
        assert len(data) == len(stat)
        for idx in range(len(stat)):
            stat[idx] = func(stat[idx], data[idx].numpy())
    return stat


def _accumulate_act_outlier_stat(stat: List[np.ndarray], data: List[tvm.nd.NDArray]):
    a_data = data[::2]
    return _accumulate_outlier_stat(stat, a_data)


def _accumulate_weight_outlier_stat(stat: List[np.ndarray], data: List[tvm.nd.NDArray]):
    # Optimization step: no need to accumulate weights for each new element in dataset since
    # weights are the same.
    if stat is not None:
        return stat
    w_data = data[1::2]
    return _accumulate_outlier_stat(stat, w_data)


def _accumulate_max_min_stat(
    a_max_stat: List[np.ndarray],
    a_min_stat: List[np.ndarray],
    w_max_stat: List[np.ndarray],
    w_min_stat: List[np.ndarray],
    data: List[tvm.nd.NDArray],
):
    """
    "data" is a list of tvm.nd.NDArray with the following structure:
      - Every first element in the sequence is the maximum values for activations corresponding to
        the output #N.
      - Every second element in the sequence is the minimum values for activations  corresponding to
        the output #N.
      - Every third element in the sequence is the maximum values for the weights corresponding to
        the output #N.
      - Every second element in the sequence is the minimum values for the weights corresponding to
        the output #N.
    """
    a_max_stat = _accumulate_outlier_stat(a_max_stat, data[::4], np.maximum)
    a_min_stat = _accumulate_outlier_stat(a_min_stat, data[1::4], np.minimum)

    if w_max_stat is None:
        w_max_stat = _accumulate_outlier_stat(w_max_stat, data[2::4], np.maximum)

    if w_min_stat is None:
        w_min_stat =  _accumulate_outlier_stat(w_min_stat, data[3::4], np.minimum)

    return a_max_stat, a_min_stat, w_max_stat, w_min_stat


def _calculate_scale_params(
    func_name: str,
    stats,
    config: Dict[str, Any],
    dev: tvm.runtime.Device,
):
    if stats[func_name] is None:
        return {}

    # scales = act_scales.pow(alpha) / weight_scales.pow(1-alpha)
    alpha = config["alpha"]
    a_stat, w_stat = stats[func_name]
    assert len(a_stat) == len(w_stat)
    idx = 0
    scale_params = {}
    for a_element, w_elemet in zip(a_stat, w_stat):
        assert a_element.shape == w_elemet.shape
        assert len(a_element.shape) == 1
        scales = np.power(a_element, alpha) / np.power(w_elemet, 1 - alpha)
        if scales.size - np.count_nonzero(scales) > 0:
            print("Warning: Smoothing: scales have zero value")
            scales = np.ones_like(scales)
            assert False, "Not supported case: please, add more elements in dataset. Otherwise, NaNs in output are possible."
        scale_params[f"sq_scale_{idx}"] = tvm.nd.array(scales, dev)
        scale_params[f"sq_scale_{idx+1}"] = tvm.nd.array(scales, dev)
        idx += 2

    return scale_params


# Quantization algorithm for activations or weights.
class QAlgo(Enum):
    PER_CHANNEL_SYM = 1,
    PER_CHANNEL_ASYM = 2,
    PER_TENSOR_SYM = 3,
    PER_TENSOR_ASYM = 4


def get_quantization_scheme(qscheme: str):
    """
    Return pair: quantization scheme for activations, quantization scheme for weights.
    """
    if qscheme == "smq_q8i8f16_0":
        return QAlgo.PER_TENSOR_SYM, QAlgo.PER_TENSOR_SYM
    elif qscheme == "smq_q8i8f16_1":
        return QAlgo.PER_TENSOR_SYM, QAlgo.PER_CHANNEL_SYM
    elif qscheme == "smq_q8i8f16_2":
        return QAlgo.PER_TENSOR_ASYM, QAlgo.PER_CHANNEL_ASYM
    else:
        assert False, f"Unknown quantization scheme: {qscheme}"
        return None, None


def _calculate_quantization_params(
    func_name: str,
    stats: Dict[str, List[np.ndarray]],
    config: Dict[str, Any],
):
    """
    Equations for asymmetric quantization (PER_TENSOR_ASYM, PER_CHANNEL_ASYM):
      scale = (MAXf - MINf)/(MAXi - MINi)  (example for "int8": scale = (MAXf - MINf)/255)
      zp = -round(MINf/scale) + MINi       (example for "int8": zp = -round(MINf/scale) - 128)

    Equations for symmetric quantization (PER_TENSOR_SYM, PER_CHANNEL_SYM):
      scale = max(abs(MAXf), abs(MINf)) / MAXi     (example for "int8": 
                                                    scale = (max(abs(MAXf), abs(MINf)))/127)
      zp = 0
    """
    if stats[func_name] is None:
        return {}

    a_dtype, w_dtype = config["adtype"], config["wdtype"]
    a_qscheme, w_qscheme = get_quantization_scheme(config["qscheme"])

    def _calculate_scale_zp(arr_max: np.ndarray, arr_min: np.ndarray, dtype: str, algo: QAlgo):
        max_value = arr_max.dtype.type(np.iinfo(dtype).max)
        min_value = arr_max.dtype.type(np.iinfo(dtype).min)
        size = arr_max.size
        if algo is QAlgo.PER_TENSOR_SYM:
            arr = np.maximum(np.abs(arr_max), np.abs(arr_min))
            scale = np.array([np.max(arr) / max_value] * size)
            zp = np.zeros_like(scale, dtype="int8")
        elif algo is QAlgo.PER_CHANNEL_SYM:
            arr = np.maximum(np.abs(arr_max), np.abs(arr_min))
            scale = arr / max_value
            zp = np.zeros_like(scale, dtype="int8")
        elif algo is QAlgo.PER_TENSOR_ASYM:
            scale = np.array([(np.max(arr_max) - np.min(arr_min)) / (max_value - min_value)] * size)
            zp = (-np.round(np.min(arr_min) / scale) + min_value).astype("int8")
        elif algo is QAlgo.PER_CHANNEL_ASYM:
            scale = (arr_max - arr_min) / (max_value - min_value)
            zp = (-np.round(arr_min / scale) + min_value).astype("int8")
        else:
            assert False, f"Unknown quantization algorithm: {algo}"
            return None, None
        return scale, zp

    idx = 0
    qparams = {}
    for a_max_element, a_min_element, w_max_element, w_min_element in zip(*stats[func_name]):
        a_scale, a_zp = _calculate_scale_zp(a_max_element, a_min_element, a_dtype, a_qscheme)
        qparams[f"{SCALE_PREFIX_NAME}{idx}"] = tvm.nd.array(a_scale, tvm.cpu(0))
        qparams[f"{ZP_PREFIX_NAME}{idx+2}"] = tvm.nd.array(a_zp, tvm.cpu(0))

        w_scale, w_zp = _calculate_scale_zp(w_max_element, w_min_element, w_dtype, w_qscheme)
        qparams[f"{SCALE_PREFIX_NAME}{idx+1}"] = tvm.nd.array(w_scale, tvm.cpu(0))
        qparams[f"{ZP_PREFIX_NAME}{idx+3}"] = tvm.nd.array(w_zp, tvm.cpu(0))
        idx += 4

    return qparams


def _smooth(
    mod: tvm.IRModule,
    params: List[tvm.nd.NDArray],
    funcs: List[str],
    dataset: List[tvm.nd.NDArray],
    config: Dict[str, Any],
):
    mod = mlc_llm.transform.SmoothQuantAnnotator()(mod)
    stat_mod = mlc_llm.transform.SmoothQuantStatCollector()(mod)
    stat_mod = mlc_llm.transform.FuseTransposeMatmul()(stat_mod)

    prefill, decode, kvc, _, _ = get_runtime_func(funcs, stat_mod)

    # Calculate max statistics
    a_stat: List[np.ndarray] = None
    w_stat: List[np.ndarray] = None

    target = tvm.target.Target.current(allow_none=False)
    for data in tqdm(dataset, desc="Smoothing"):
        # Create KV-cache
        kv_caches = kvc()
        num_tokens = data.shape[1]
        seq_len_shape = tvm.runtime.ShapeTuple([num_tokens])

        # Run Encoder and update statistics for activations/weights
        (logits, kv_caches), outputs = prefill(data, seq_len_shape, kv_caches, *params)
        a_stat = _accumulate_act_outlier_stat(a_stat, outputs)
        w_stat = _accumulate_weight_outlier_stat(w_stat, outputs)

        # Run Decoder and update statistics for activations/weights
        for _ in range(config["decoder_invoke_num"]):
            # TODO: support softmax with temperature.
            logits_max = np.argmax(logits.numpy(), axis=-1).astype("int32")
            if logits_max[0] in config["stop_tokens"]:
                break
            next_token = tvm.nd.array(logits_max, device=tvm.device(target.kind.default_keys[0]))
            num_tokens += logits_max.shape[1]
            seq_len_shape = tvm.runtime.ShapeTuple([num_tokens])
            (logits, kv_caches), outputs = decode(next_token, seq_len_shape, kv_caches, *params)
            a_stat = _accumulate_act_outlier_stat(a_stat, outputs)
            w_stat = _accumulate_weight_outlier_stat(w_stat, outputs)

    # Use the same statistics for "prefill"/"decode"
    stat = dict.fromkeys(funcs)
    stat["prefill"] = (a_stat, w_stat)
    stat["decode"] = (a_stat, w_stat)
    for fname in funcs:
        scale_params = _calculate_scale_params(fname, stat, config, tvm.cpu(0))
        mod = relax.transform.BindParams(fname, scale_params)(mod)
    return mod


def _calibrate(
    mod: tvm.IRModule,
    params: List[tvm.nd.NDArray],
    funcs: List[str],
    dataset: List[tvm.nd.NDArray],
    config: Dict[str, Any],
):
    qscheme: str = config["qscheme"]
    mod = mlc_llm.transform.SmoothQuantAnnotator(qscheme)(mod)
    stat_mod = mlc_llm.transform.SmoothQuantStatCollector()(mod)
    stat_mod = mlc_llm.transform.FuseTransposeMatmul()(stat_mod)

    prefill, decode, kvc, _, _ = get_runtime_func(funcs, stat_mod)

    # Calculate max statistics
    a_max_stat: List[np.ndarray] = None
    a_min_stat: List[np.ndarray] = None
    w_max_stat: List[np.ndarray] = None
    w_min_stat: List[np.ndarray] = None

    target = tvm.target.Target.current(allow_none=False)
    for data in tqdm(dataset, desc="Calibration"):
        # Create KV-cache
        kv_caches = kvc()
        num_tokens = data.shape[1]
        seq_len_shape = tvm.runtime.ShapeTuple([num_tokens])

        # Run Encoder and update statistics for activations/weights
        (logits, kv_caches), outputs = prefill(data, seq_len_shape, kv_caches, *params)
        a_max_stat, a_min_stat, w_max_stat, w_min_stat = _accumulate_max_min_stat(
            a_max_stat, a_min_stat, w_max_stat, w_min_stat, outputs
        )

        # Run Decoder and update statistics for activations/weights
        for _ in range(config["decoder_invoke_num"]):
            # TODO: support softmax with temperature.
            logits_max = np.argmax(logits.numpy(), axis=-1).astype("int32")
            if logits_max[0] in config["stop_tokens"]:
                break
            next_token = tvm.nd.array(logits_max, device=tvm.device(target.kind.default_keys[0]))
            num_tokens += logits_max.shape[1]
            seq_len_shape = tvm.runtime.ShapeTuple([num_tokens])
            (logits, kv_caches), outputs = decode(next_token, seq_len_shape, kv_caches, *params)
            a_max_stat, a_min_stat, w_max_stat, w_min_stat = _accumulate_max_min_stat(
                a_max_stat, a_min_stat, w_max_stat, w_min_stat, outputs
            )

    # Use the same statistics for "prefill"/"decode"
    stat = dict.fromkeys(funcs)
    stat["prefill"] = (a_max_stat, a_min_stat, w_max_stat, w_min_stat)
    stat["decode"] = (a_max_stat, a_min_stat, w_max_stat, w_min_stat)
    for fname in funcs:
        scale_params = _calculate_quantization_params(fname, stat, config)
        mod = relax.transform.BindParams(fname, scale_params)(mod)

    mod = mlc_llm.transform.SmoothQuantLegalizer(config["adtype"], config["wdtype"])(mod)
    mod = relax.transform.DeadCodeElimination(funcs)(mod)
    return mod


def smoothquant(
    args: argparse.Namespace,
    mod: tvm.IRModule,
    cpu_params: List[tvm.nd.NDArray],
    model_names: List[str],
):
    target = args.target
    smq_device = tvm.device(target.kind.default_keys[0])
    params = to_device(cpu_params, smq_device)
    # Free memory on the host:
    cpu_params.clear()

    dataset, stop_tokens = _get_dataset(args.dataset, args.artifact_path, device=smq_device)
    smq_config: Dict[str, Any] = {}
    smq_config["decoder_invoke_num"] = 5
    smq_config["alpha"] = 0.5
    smq_config["stop_tokens"] = stop_tokens
    smq_config["adtype"] = "int8"
    smq_config["wdtype"] = "int8"
    smq_config["qscheme"] = args.quantization.name
    with target:
        print("[SmoothQuant] Run smoothing...")
        mod = _smooth(mod, params, model_names, dataset, smq_config)
        print("[SmoothQuant] Run calibration and quantization...")
        mod = _calibrate(mod, params, model_names, dataset, smq_config)
        print("[SmoothQuant] Smoothing and calibration were done!")

    mod = mlc_llm.transform.SmoothQuantStopLiftParamsOptimizer()(mod)
    mod = relax.transform.LiftTransformParams()(mod)
    mod = relax.transform.BundleModelParams()(mod)
    mod_transform, mod_deploy = utils.split_transform_deploy_mod(mod, model_names)
    new_params = smoothquant_transform_params(mod_transform, params, target)

    # Free memory on device:
    params.clear()

    return mod_deploy, new_params


def smoothquant_transform_params(
    mod_transform: tvm.IRModule,
    params: List[tvm.nd.NDArray],
    target: tvm.target.Target,
) -> List[tvm.nd.NDArray]:
    mod_transform = relax.transform.ToNonDataflow()(mod_transform)
    mod_transform = relax.transform.LazyTransformParams()(mod_transform)
    mod_transform = tvm.relax.transform.LegalizeOps()(mod_transform)

    smq_device = tvm.device(target.kind.default_keys[0])
    new_params: List[tvm.nd.NDArray] = []

    @tvm.register_func("get_item", override=True)
    def get_item(i):
        assert params[i] is not None
        return tvm.nd.array(params[i], device=smq_device)

    @tvm.register_func("set_item", override=True)
    def set_item(i, value):
        if len(new_params) <= i:
            new_params.extend([None for _ in range(i - len(new_params) + 1)])
        new_params[i] = tvm.nd.array(value, device=tvm.cpu())

    if target.kind.name != "llvm":
        with tvm.target.Target(target):
            mod_transform = tvm.tir.transform.DefaultGPUSchedule()(mod_transform)

    assert "decode_transform_params" in [gv.name_hint for gv in mod_transform.get_global_vars()]
    ex = relax.build(mod_transform, target=target)
    vm = relax.vm.VirtualMachine(ex, smq_device)
    vm["decode_transform_params"]()
    return new_params


def _prepare_dummy_dataset(artifact_path: str):
    """
    Return path to file with dummy dataset. This dataset consists of 4 simple questions.
    "train", "test" and "validation" datasets use the same dummy file.
    """
    file_name = "dummy_calibration_dataset.txt"
    file_path = os.path.join(artifact_path, file_name)
    data_files = {"train": file_path, "test": file_path, "validation": file_path}
    if os.path.exists(file_path):
        return data_files

    prompts = [
        "The capital of Canada is",
        "2+2=?",
        "What is the capital of France?",
        "Who is the president of the USA?",
    ]
    f = open(file_path, "w+")
    for prompt in prompts:
        f.write(prompt + "\n")
    return data_files


def _get_dataset(name: str, artifact_path: str, device: tvm.runtime.Device):
    print("[SmoothQuant] Starting to initialize tokenizer...")
    tokenizer_path = os.path.join(artifact_path, "params")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    print("[SmoothQuant] Initialization of tokenizer was completed...")

    if name not in dataset_list:
        raise ValueError(f"Dataset {name} is not supported")
    config_name = None
    split = "train"
    if name == "piqa":
        data_files = None
        text_name = "goal"
    elif name == "gsm8k":
        data_files = None
        text_name = "question"
        config_name = "main"
        split = "test[:10%]"
    else:
        # Dummy dataset consisting of 4 simple questions.
        name = text_name = "text"
        data_files = _prepare_dummy_dataset(artifact_path)
        config_name = None
    dataset = load_dataset(name, name=config_name, data_files=data_files, split=split)
    calibration_dataset = []
    for record in dataset:
        data = tokenizer(record[text_name], return_tensors="np")
        calibration_dataset.append(tvm.nd.array(data["input_ids"].astype("int32"), device=device))
    stop_tokens = ([tokenizer.eos_token_id])

    return calibration_dataset, stop_tokens


def debug_save_stat(stat: Union[Dict[str, tvm.nd.NDArray], List[tvm.nd.NDArray]], name: str):
    folder_name = "dump_npy"
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    if isinstance(stat, dict):
        for key, value in stat.items():
            with open(os.path.join(folder_name, f"{name}_{key}.npy"), "wb") as f:
                np.save(f, value.numpy())
    else:
        assert isinstance(stat, list)
        for idx, element in enumerate(stat):
            with open(os.path.join(folder_name, f"{name}_{idx}.npy"), "wb") as f:
                np.save(f, element)
