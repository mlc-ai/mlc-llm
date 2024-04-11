import numpy as np
from enum import Enum
from typing import List, Dict, Any, Callable

import tvm

from slm.compiler_pass.smoothquant import get_scale_param_name, get_zp_param_name, SMOOTH_SUFFIX_NAME, QUANT_SUFFIX_NAME

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

    alpha = config["alpha"]
    a_stat, w_stat = stats[func_name]
    assert len(a_stat) == len(w_stat)
    idx = 0
    scale_params = {}
    for a_element, w_element in zip(a_stat, w_stat):
        assert a_element.shape == w_element.shape
        assert len(a_element.shape) == 1
        scales = np.power(np.abs(a_element), alpha) / np.power(np.abs(w_element), 1 - alpha)
        if scales.size - np.count_nonzero(scales) > 0:
            print("Warning: Smoothing: scales have zero value")
            scales = np.where(scales == 0.0, 1.0, scales)
        scale_params[get_scale_param_name(idx, SMOOTH_SUFFIX_NAME)] = tvm.nd.array(scales, dev)
        scale_params[get_scale_param_name(idx+1, SMOOTH_SUFFIX_NAME)] = tvm.nd.array(scales, dev)
        idx += 2

    return scale_params


# Quantization algorithm for activations or weights.
class QAlgo(Enum):
    PER_CHANNEL_SYM = 1,
    PER_CHANNEL_ASYM = 2,
    PER_TENSOR_SYM = 3,
    PER_TENSOR_ASYM = 4


def get_quantization_scheme(qscheme: str):

    """Return pair: quantization scheme for activations, quantization scheme for weights.
    """

    if qscheme == "smq_q8i8f16_0":
        return QAlgo.PER_TENSOR_SYM, QAlgo.PER_TENSOR_SYM
    elif qscheme == "smq_q8i8f16_1":
        return QAlgo.PER_TENSOR_SYM, QAlgo.PER_CHANNEL_SYM
    elif qscheme in {"smq_q8i8f16_2", "smq_q8i8f32_2"}:
        return QAlgo.PER_TENSOR_ASYM, QAlgo.PER_CHANNEL_ASYM
    elif qscheme in {"smq_e4m3_float8_0", "smq_e5m2_float8_0"}:
        return QAlgo.PER_TENSOR_SYM, QAlgo.PER_TENSOR_SYM
    elif qscheme  in {"smq_e4m3_float8_1", "smq_e5m2_float8_1"}:
        return QAlgo.PER_TENSOR_SYM, QAlgo.PER_CHANNEL_SYM
    elif qscheme in {"smq_e4m3_float8_2", "smq_e5m2_float8_2"}:
        return QAlgo.PER_TENSOR_ASYM, QAlgo.PER_CHANNEL_ASYM
    else:
        assert False, f"Unknown quantization scheme: {qscheme}"
        return None, None


def _calculate_quantization_params(
    func_name: str,
    stats: Dict[str, List[np.ndarray]],
    config: Dict[str, Any],
):

    """Equations for asymmetric quantization (PER_TENSOR_ASYM, PER_CHANNEL_ASYM):
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
        if dtype.startswith("int"):
            max_value = arr_max.dtype.type(np.iinfo(dtype).max)
            min_value = arr_max.dtype.type(np.iinfo(dtype).min)
            zp_type = dtype
        elif dtype == "e4m3_float8": # based on https://arxiv.org/pdf/2209.05433.pdf
            max_value = arr_max.dtype.type(448)
            min_value = arr_max.dtype.type(-448)
            zp_type = "float16"
        elif dtype == "e5m2_float8":
            max_value = arr_max.dtype.type(57344)
            min_value = arr_max.dtype.type(-57344)
            zp_type = "float16"
        size = arr_max.size
        if algo is QAlgo.PER_TENSOR_SYM:
            arr = np.maximum(np.abs(arr_max), np.abs(arr_min))
            scale = np.array([np.max(arr) / max_value] * size)
            zp = np.zeros_like(scale, dtype=zp_type)
        elif algo is QAlgo.PER_CHANNEL_SYM:
            arr = np.maximum(np.abs(arr_max), np.abs(arr_min))
            scale = arr / max_value
            zp = np.zeros_like(scale, dtype=zp_type)
        elif algo is QAlgo.PER_TENSOR_ASYM:
            scale = np.array([(np.max(arr_max) - np.min(arr_min)) / (max_value - min_value)] * size)
            zp = (-np.round(np.min(arr_min) / scale) + min_value).astype(zp_type)
        elif algo is QAlgo.PER_CHANNEL_ASYM:
            scale = (arr_max - arr_min) / (max_value - min_value)
            zp = (-np.round(arr_min / scale) + min_value).astype(zp_type)
        else:
            assert False, f"Unknown quantization algorithm: {algo}"
            return None, None
        return scale, zp

    idx = 0
    qparams = {}
    for a_max_element, a_min_element, w_max_element, w_min_element in zip(*stats[func_name]):
        a_scale, a_zp = _calculate_scale_zp(a_max_element, a_min_element, a_dtype, a_qscheme)
        qparams[get_scale_param_name(idx, QUANT_SUFFIX_NAME)] = tvm.nd.array(a_scale, tvm.cpu(0))
        qparams[get_zp_param_name(idx+2)] = tvm.nd.array(a_zp, tvm.cpu(0))

        w_scale, w_zp = _calculate_scale_zp(w_max_element, w_min_element, w_dtype, w_qscheme)
        qparams[get_scale_param_name(idx+1, QUANT_SUFFIX_NAME)] = tvm.nd.array(w_scale, tvm.cpu(0))
        qparams[get_zp_param_name(idx+3)] = tvm.nd.array(w_zp, tvm.cpu(0))
        idx += 4

    return qparams
