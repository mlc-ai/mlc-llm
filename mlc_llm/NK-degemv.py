# pylint: disable=missing-docstring
import os
from typing import Callable, Dict, List, Optional

import numpy as np

import tvm

import tvm.testing
from tvm import ir, te, tir, dlight
from tvm.contrib import nvcc, rpc, utils, ndk
from tvm.script import tir as T, ir as I

############ CUDA
TARGET = tvm.target.Target("nvidia/geforce-rtx-2070")
DEVICE = tvm.cuda(0)
LOAD_V_SHARED = True
LOAD_V_VEC = 8
TILE_N = False

############ Metal
# TARGET = tvm.target.Target("metal")
# DEVICE = tvm.metal(0)
# LOAD_V_SHARED = True
# LOAD_V_VEC = 4
# TILE_N = False

############ Mali
# tracker_host = "192.168.10.1"
# tracker_port = 9191
# key = "orangepi"

# TARGET = tvm.target.Target(
#     "opencl -device=mali", host="llvm -mtriple=aarch64-linux-gnu"
# )

# tracker = rpc.connect_tracker(tracker_host, tracker_port)
# remote = tracker.request(key, priority=0, session_timeout=0)
# DEVICE = remote.cl(0)
# LOAD_V_SHARED = False
# TILE_N = True

############ Android
# # Set to be address of tvm proxy.
# tracker_host = "0.0.0.0"
# tracker_port = 9090
# key = "android"

# # Change target configuration.
# # Run `adb shell cat /proc/cpuinfo` to find the arch.
# arch = "arm64"
# target = "llvm -mtriple=%s-linux-android" % arch
# TARGET = tvm.target.Target("opencl", host=target)

# tracker = rpc.connect_tracker(tracker_host, tracker_port)
# remote = tracker.request(key, priority=0, session_timeout=0)
# DEVICE = remote.cl(0)
# LOAD_V_SHARED = False
# TILE_N = True

############

N = 12288
K = 4096
# N = 15360
# K = 5120

cur_best = 1e6
cur_best_dict = None

# fmt: off

@T.prim_func
def NK_degemv(
    A_q: T.Buffer((T.int64(N), T.int64(K // 8)), "uint32"), 
    A_scale: T.Buffer((T.int64(N), T.int64(K // 32)), "float16"), 
    V: T.Buffer((T.int64(K)), "float16"), 
    C: T.Buffer((T.int64(N)), "float16")
):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    A = T.alloc_buffer((T.int64(N), T.int64(K)), "float16")
    for i, j in T.grid(T.int64(N), T.int64(K)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(A_q[v_i, v_j // T.int64(8)], A_scale[v_i, v_j // T.int64(32)])
            T.writes(A[v_i, v_j])
            A[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(A_q[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * A_scale[v_i, v_j // T.int64(32)]
    for i2, k in T.grid(T.int64(N), T.int64(K)):
        with T.block("gemv"):
            v_i2, v_k = T.axis.remap("SR", [i2, k])
            T.reads(V[v_k], A[v_i2, v_k])
            T.writes(C[v_i2])
            with T.init():
                C[v_i2] = T.float16(0)
            C[v_i2] = C[v_i2] + V[v_k] * A[v_i2, v_k]


def get_NK_degemv_n(n):
    @T.prim_func
    def NK_degemv_n(
        A_q: T.Buffer((T.int64(N // n), T.int64(K // 8), T.int64(n)), "uint32"), 
        A_scale: T.Buffer((T.int64(N // n), T.int64(K // 32), T.int64(n)), "float16"), 
        V: T.Buffer((T.int64(K)), "float16"), 
        C: T.Buffer((T.int64(N)), "float16")
    ):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        A = T.alloc_buffer((T.int64(N), T.int64(K)), "float16")
        for i, j in T.grid(T.int64(N), T.int64(K)):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(A_q[v_i // T.int64(n), v_j // T.int64(8), v_i % T.int64(n)], A_scale[v_i // T.int64(n), v_j // T.int64(32), v_i % T.int64(n)])
                T.writes(A[v_i, v_j])
                A[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(A_q[v_i // T.int64(n), v_j // T.int64(8), v_i % T.int64(n)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * A_scale[v_i // T.int64(n), v_j // T.int64(32), v_i % T.int64(n)]
        for i2, k in T.grid(T.int64(N), T.int64(K)):
            with T.block("gemv"):
                v_i2, v_k = T.axis.remap("SR", [i2, k])
                T.reads(V[v_k], A[v_i2, v_k])
                T.writes(C[v_i2])
                with T.init():
                    C[v_i2] = T.float16(0)
                C[v_i2] = C[v_i2] + V[v_k] * A[v_i2, v_k]
    return NK_degemv_n


def get_NK_degemv_nk(n, k):
    @T.prim_func
    def NK_degemv_nk(
        A_q: T.Buffer((T.int64(N // n), T.int64(K // 8 // k), T.int64(n), T.int64(k)), "uint32"), 
        A_scale: T.Buffer((T.int64(N // n), T.int64(K // 32), T.int64(n)), "float16"), 
        V: T.Buffer((T.int64(K)), "float16"), 
        C: T.Buffer((T.int64(N)), "float16")
    ):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        A = T.alloc_buffer((T.int64(N), T.int64(K)), "float16")
        for i, j in T.grid(T.int64(N), T.int64(K)):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(A_q[v_i // T.int64(n), v_j // T.int64(8) // T.int64(k), v_i % T.int64(n), v_j // T.int64(8) % T.int64(k)], A_scale[v_i // T.int64(n), v_j // T.int64(32), v_i % T.int64(n)])
                T.writes(A[v_i, v_j])
                A[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(A_q[v_i // T.int64(n), v_j // T.int64(8) // T.int64(k), v_i % T.int64(n), v_j // T.int64(8) % T.int64(k)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * A_scale[v_i // T.int64(n), v_j // T.int64(32), v_i % T.int64(n)]
        for i2, r in T.grid(T.int64(N), T.int64(K)):
            with T.block("gemv"):
                v_i2, v_k = T.axis.remap("SR", [i2, r])
                T.reads(V[v_k], A[v_i2, v_k])
                T.writes(C[v_i2])
                with T.init():
                    C[v_i2] = T.float16(0)
                C[v_i2] = C[v_i2] + V[v_k] * A[v_i2, v_k]
    return NK_degemv_nk

# fmt: on


def prepare_args(func: tir.PrimFunc, var_dict: Dict[str, int]):
    np.random.seed(0)
    args: List[np.ndarray] = []
    analyzer = tvm.arith.Analyzer()
    total_bytes = 0
    for param in func.params:
        buffer = func.buffer_map[param]
        shape = []
        for dim in buffer.shape:
            if isinstance(dim, tir.IntImm):
                shape.append(dim.value)
            elif isinstance(dim, tir.Var):
                assert dim.name in var_dict
                value = var_dict[dim.name]
                shape.append(value)
                analyzer.bind(dim, value)
            else:
                raise ValueError(f"Unknown shape: {buffer.shape}")
        if buffer.dtype == "uint32":
            np_array = np.random.randint(0, 2**16, size=shape).astype(buffer.dtype)
        else:
            np_array = np.random.uniform(high=0.01, size=shape).astype(buffer.dtype)
        total_bytes += np_array.size * np_array.itemsize
        tvm_array = tvm.nd.array(np_array, DEVICE)
        args.append(tvm_array)
    return args, total_bytes


def build_and_measure(func: tir.PrimFunc, args, total_bytes, config, run_only=False):
    rt_mod = tvm.build(func, target=TARGET)
    ################# Android or Mali
    temp = utils.tempdir()
    path_dso_cl = temp.relpath("dev_lib_cl.so")
    rt_mod.export_library(path_dso_cl, ndk.create_shared)
    remote.upload(path_dso_cl)
    rt_mod = remote.load_module("dev_lib_cl.so")
    ################# Android or Mali
    rt_mod(*args)
    ret = args[-1]
    if not run_only:
        DEVICE.sync()
        time_eval = rt_mod.time_evaluator(
            rt_mod.entry_name,
            DEVICE,
            # number=20,
            # repeat=3,
            number=1,
            repeat=100,
            cache_flush_bytes=256 * 10**6,
        )
        DEVICE.sync()
        time = time_eval(*args).mean * 1e3
        DEVICE.sync()
        bandwidth = total_bytes / time / (1024**2)

        global cur_best, cur_best_dict
        if time < cur_best and config is not None:
            cur_best = time
            cur_best_dict = config
        print(
            f"Time (ms): {time:.6f}",
            f"Total Bytes (MB): {total_bytes / (1024**2):.6f}",
            f"Memory (GB/s): {bandwidth:.6f}",
            sep="\t",
        )
        print(
            f"Best time (ms): {cur_best:.6f}",
            f"Best Memory (GB/s): {total_bytes / cur_best / (1024**2):.6f}",
            f"Best config: {cur_best_dict}",
            sep="\t",
        )
    return ret


def export_source(mod):
    lib = tvm.build(mod, target=TARGET)
    source = lib.imported_modules[0].get_source()
    # remove content before extern "C"
    print(source[source.index('extern "C"') :])
    # with open("./gemv.cu", "w") as f:
    #     f.write(source)


def get_max_factor(n, factors):
    for factor in factors[::-1]:
        if n % factor == 0:
            return factor


def schedule1(ret):
    # fmt: off
    # vector load over N, vector compute over K
    def apply(mod):
        # sanity check
        if TILE_S % VEC_LOAD != 0 or TILE_R % 8 != 0:
            return None
        if VEC_LOAD not in [1, 2, 4, 8] or VEC_C not in [1, 2, 4, 8]:
            return None

        # if tx is along s, tile_s should equal to vec_load
        # if tx is along r, tile_r should equal to 8
        # both are required by coalesced load
        if TAG_S == "threadIdx.x" and TILE_S != VEC_LOAD:
            return None
        if TAG_R == "threadIdx.x" and TILE_R != 8:
            return None
    
        sch = tir.Schedule(mod)
        
        decode = sch.get_block(name="decode", func_name="main")
        gemv = sch.get_block(name="gemv", func_name="main")
        
        sch.compute_inline(decode)
        # rfactor: reduce to tx * vec_c
        s, r = sch.get_loops(block=gemv)
        bx, ts, tile_s = sch.split(s, factors=[None, TS, TILE_S], preserve_unit_iters=True)
        r, tr, tile_r_vec_n, vec_c = sch.split(r, factors=[None, TR, TILE_R // VEC_C, VEC_C], preserve_unit_iters=True)
        sch.reorder(r, tile_r_vec_n, tr, vec_c)
        tr_vec_c = sch.fuse(tr, vec_c)
        rf = sch.rfactor(tr_vec_c, 0)

        # rfactor: reduce to tx
        bx, ts, tile_s, tr_vec_c = sch.get_loops(block=gemv)
        tr, vec_c = sch.split(tr_vec_c, factors=[TR, None], preserve_unit_iters=True)
        rf2 = sch.rfactor(tr, 0)
        
        # bind, vectorize compute
        bx, ts, tile_s, r, tile_r_vec_n, tr_vec_c = sch.get_loops(block=rf)
        tr, vec_c = sch.split(tr_vec_c, factors=[TR, None], preserve_unit_iters=True)
        sch.reorder(bx, ts, tr, r, tile_s, tile_r_vec_n, vec_c)
        sch.bind(bx, "blockIdx.x")
        sch.bind(ts, TAG_S)
        sch.bind(tr, TAG_R)
        sch.vectorize(vec_c)

        # vectorize load A
        Aq_local = sch.cache_read(rf, read_buffer_index=1, storage_scope="local")
        sch.compute_at(Aq_local, r, preserve_unit_loops=True) 
        # ^ should have shape (tile_s, tile_r // 8, 1, 1), since tile_s <= n, tile_r <= k * 8
        l_s, l_r = sch.get_loops(block=Aq_local)[-2:]
        l_s, vec_load = sch.split(l_s, factors=[None, VEC_LOAD], preserve_unit_iters=True)
        sch.reorder(l_s, l_r, vec_load)
        sch.vectorize(vec_load)
        
        # load vector into shared memory, shape should be the whole vector
        if LOAD_V_SHARED:
            V_shared = sch.cache_read(rf, read_buffer_index=0, storage_scope="shared")
            sch.compute_at(V_shared, tr, preserve_unit_loops=True)
            l = sch.get_loops(block=V_shared)[-1]
            if TAG_R == "threadIdx.x":
                _, ty, tx, vec = sch.split(l, factors=[None, TS, TR, LOAD_V_VEC], preserve_unit_iters=True)
            else:
                _, ty, tx, vec = sch.split(l, factors=[None, TR, TS, LOAD_V_VEC], preserve_unit_iters=True)
            sch.bind(ty, "threadIdx.y")
            sch.bind(tx, "threadIdx.x")
            sch.vectorize(vec)
        
        # reduce tile_s * tr * vec to tile_s * tr
        sch.reverse_compute_at(rf2, loop=bx, preserve_unit_loops=True)
        tr, vec_c, ts_tile_s = sch.get_loops(block=rf2)[-3:]
        ts, tile_s = sch.split(ts_tile_s, factors=[TS, None], preserve_unit_iters=True)
        tile_s, vec_s = sch.split(tile_s, factors=[None, get_max_factor(TILE_S, [1, 2, 4, 8])], preserve_unit_iters=True)
        sch.reorder(ts, tr, tile_s, vec_s, vec_c)
        sch.bind(ts, TAG_S)
        sch.bind(tr, TAG_R)
        sch.vectorize(vec_s)
        
        # reduce tile_s * tr to tile_s
        sch.reverse_compute_at(gemv, loop=bx, preserve_unit_loops=True)
        tr, ts_tile_s = sch.get_loops(block=gemv)[-2:]
        ts, tile_s = sch.split(ts_tile_s, factors=[TS, None], preserve_unit_iters=True)
        sch.reorder(tile_s, ts, tr)
        sch.bind(ts, TAG_S)
        sch.bind(tr, TAG_R)

        if TR == 1:
            tile_s, vec_s = sch.split(tile_s, factors=[None, get_max_factor(TILE_S, [1, 2, 4, 8])], preserve_unit_iters=True)
            sch.vectorize(vec_s)
        
        sch.decompose_reduction(rf, loop=sch.get_loops(block=rf)[3])
        sch.decompose_reduction(rf2, loop=sch.get_loops(block=rf2)[-1])
        
        sch.set_scope(rf, buffer_index=0, storage_scope="local")
        sch.set_scope(rf2, buffer_index=0, storage_scope="local")

        unroll_factor = 256

        sch.annotate(block_or_loop=sch.get_loops(rf)[3], ann_key="pragma_auto_unroll_max_step", ann_val=unroll_factor)
        sch.annotate(block_or_loop=sch.get_loops(rf)[3], ann_key="pragma_unroll_explicit", ann_val=1)

        sch.annotate(block_or_loop=sch.get_loops(rf2)[3], ann_key="pragma_auto_unroll_max_step", ann_val=unroll_factor)
        sch.annotate(block_or_loop=sch.get_loops(rf2)[3], ann_key="pragma_unroll_explicit", ann_val=1)

        if LOAD_V_SHARED:
            sch.annotate(block_or_loop=sch.get_loops(V_shared)[-4], ann_key="pragma_unroll_explicit", ann_val=unroll_factor)
            sch.annotate(block_or_loop=sch.get_loops(V_shared)[-4], ann_key="pragma_vectorize", ann_val=1)

        return sch

    func_dict = dict()
    arg_dict = dict()
    # fmt: on
    for n in [1, 2, 4, 8, 16, 64] if TILE_N else [1]:
        for k in [1]:
            func_dict[(n, k)] = get_NK_degemv_nk(n, k)
            arg_dict[(n, k)] = prepare_args(func_dict[(n, k)], {"n": n})
    for TAG_S, TAG_R in [
        ("threadIdx.x", "threadIdx.y"),
        ("threadIdx.y", "threadIdx.x"),
    ]:
        for VEC_LOAD in [1, 2, 4]:
            for VEC_C in [4, 2, 1]:
                for TILE_S in [1, 2, 4]:
                    for TILE_R in [1]:
                        TILE_R = TILE_R * 8
                        for all_thread in [512, 256, 128, 64, 32]:
                            for TR in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
                                TS = all_thread // TR
                                if TS <= 0 or TR <= 0:
                                    continue
                                if TILE_S > n or TILE_R > k * 8:
                                    continue
                                for (n, k), func in func_dict.items():
                                    if N % n != 0 or K % k != 0:
                                        continue
                                    sch = apply(func)
                                    if sch is None:
                                        continue
                                    try:
                                        print("====")
                                        print(
                                            f"schedule 1: TAG_S={TAG_S}",
                                            f"TAG_R={TAG_R}",
                                            f"vec_load={VEC_LOAD}",
                                            f"vec_c={VEC_C}",
                                            f"tile_s={TILE_S}",
                                            f"tile_r={TILE_R}",
                                            f"tr={TR}",
                                            f"ts={TS}",
                                            f"n={n}",
                                            f"k={k}",
                                            sep="\t",
                                        )
                                        # print(sch.mod.script())
                                        ret_cur = build_and_measure(
                                            sch.mod["main"],
                                            arg_dict[(n, k)][0],
                                            arg_dict[(n, k)][1],
                                            config={
                                                "TAG_S": TAG_S,
                                                "TAG_R": TAG_R,
                                                "VEC_LOAD": VEC_LOAD,
                                                "VEC_C": VEC_C,
                                                "TILE_S": TILE_S,
                                                "TILE_R": TILE_R,
                                                "TR": TR,
                                                "TS": TS,
                                                "n": n,
                                                "k": k,
                                            },
                                        )
                                        if not TILE_N:
                                            tvm.testing.assert_allclose(
                                                ret.numpy(),
                                                ret_cur.numpy(),
                                                rtol=5e-2,
                                                atol=5e-2,
                                            )
                                        # export_source(sch.mod["main"])
                                    except Exception as e:
                                        print("Error", e)


def schedule2():
    # vector load over N, vector compute over N
    pass


def schedule3():
    # vector load over K, vector compute over K
    pass


def schedule4():
    # vector load over K, vector compute over N
    pass


def main():
    dlight_sch = dlight.gpu.GEMV().apply(NK_degemv, TARGET, False)
    dlight_mod = dlight_sch.mod
    # dlight_mod.show(black_format=False)
    print("dlight:")
    args = prepare_args(dlight_mod["main"], {"n": 256})
    ret = build_and_measure(dlight_mod["main"], *args, None, run_only=False)

    # schedule 1 vector load over N, vector compute over K
    schedule1(ret)

    # schedule 2, vector load over N, vector compute over N
    schedule2()

    # schedule 3, vector load over K, vector compute over K
    schedule3()

    # schedule 4, vector load over K, vector compute over N
    schedule4()


if __name__ == "__main__":
    main()
