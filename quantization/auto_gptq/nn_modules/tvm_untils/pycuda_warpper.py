from typing import Any
import torch
import pycuda
import pycuda.autoprimaryctx
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from .workloads import get_gemm_workloads, get_gemv_workloads, _apply_gemm_schedule, _apply_gemv_schedule, _apply_dynamic_gemm_schedule
import numpy as np
import os
import nni
from nni.experiment import Experiment
from .nni_database import NNIDatabase
import time

nni_database_path = '.nnidatabase'

class PortPool(object):
    def __init__(self, init_port: int = 8080):
        self._port = init_port
        pass

    def get_port(self):
        self._port += 1
        return self._port


class TensorHolder(pycuda.driver.PointerHolderBase):
    def __init__(self, t):
        super(TensorHolder, self).__init__()
        self.t = t
        self.gpudata = t.data_ptr()

    def get_pointer(self):
        return self.t.data_ptr()


class TVMExecutable(object):
    def __init__(self, src, name):
        __doc__ = 'Initialize FastMLM object'
        super(TVMExecutable, self).__init__()
        self.source_code: str = src
        self.func_name: str = name
        self.kernel_func = self._get_kernel(self.source_code, self.func_name)

    def __call__(self, input, qweight, output, scales, zeros, M, grid: tuple, block: tuple) -> Any:
        if 1 == M:
            self.kernel_func(TensorHolder(input), TensorHolder(qweight), TensorHolder(
                scales), TensorHolder(zeros), TensorHolder(output), grid=grid, block=block)
        else:
            m = np.int32(M)
            self.kernel_func(TensorHolder(input), TensorHolder(qweight), TensorHolder(
                scales), TensorHolder(zeros), TensorHolder(output), m, grid=grid, block=block)

    def _get_kernel(self, src_code, name):
        src = torch.cuda.ByteTensor(8)
        mod = SourceModule(src_code, no_extern_c=True)
        return mod.get_function(name)


class TVMHandler(object):
    def __init__(self, n: int, k: int, bits: int, group_size: int, load_from_cache:bool = False):
        __doc__ = 'Initialize FastMLM object'
        super(TVMHandler, self).__init__()
        self.k = k
        self.n = n
        self.bits = bits
        self.group_size = group_size
        self.trails = 1000
        self.port_pool = PortPool(8080)
        self.working_directory = nni_database_path
        self.m_candidates = [1, 16, 32, 64, 128, 256]
        self._apply_gemm_schedule = _apply_gemm_schedule
        self._apply_gemv_schedule = _apply_gemv_schedule
        
        self.configurations = {
            'm1':
                {
                    'num_warps': 4
                },
            'm16': {
                "block_row_warps": 1,
                "block_col_warps": 4,
                "BM": 16,
                "BN": 64,
                "BK": 64,
                "raster": 0,
                "stage": 3
            },
            'm32': {
                "block_row_warps": 2,
                "block_col_warps": 4,
                "BM": 32,
                "BN": 64,
                "BK": 64,
                "raster": 0,
                "stage": 2
                },
            'm64': {
                "block_row_warps": 2,
                "block_col_warps": 4,
                "BM": 64,
                "BN": 64,
                "BK": 64,
                "raster": 0,
                "stage": 2
            },
            'm128': {
                "block_row_warps": 1,
                "block_col_warps": 2,
                "BM": 64,
                "BN": 128,
                "BK": 32,
                "raster": 0,
                "stage": 3
            },
            'm256': {
                "block_row_warps": 2,
                "block_col_warps": 2,
                "BM": 128,
                "BN": 256,
                "BK": 32,
                "raster": 0,
                "stage": 1
            }
        }

        if not load_from_cache:
            self._tune(n, k, bits)
            for m in self.m_candidates:
                if m == 1:
                    self.m1: TVMExecutable = self._get_executable_m1(bits, n, k)
                else:
                    setattr(self, f'm{m}', self._get_executable_mx(
                    bits, m, n, k))

    def __call__(self, input, qweight, output, scales, zeros) -> Any:
        assert len(output.shape) >= 2, "output should be larger than 2D"
        M = 1
        for i in range(len(input.shape) - 1):
            M *= input.shape[i]
        N = output.shape[-1]
        if M == 1:
            m1_config = self.configurations['m1']
            block = (32, m1_config['num_warps'], 1)
            grid = (N // m1_config['num_warps'], 1, 1)
            self.m1(input, qweight, output, scales,
                    zeros, M, block=block, grid=grid)
        elif 1< M <= 16:
            mx_config = self.configurations['m16']
            block = (32, mx_config['block_row_warps'],
                     mx_config['block_col_warps'])
            MPAD = (M + mx_config['BM'] - 1) // mx_config['BM'] * mx_config['BM']
            grid = (N // mx_config['BN'], MPAD // mx_config['BM'], 1)
            self.m16(input, qweight, output, scales,
                     zeros, M, block=block, grid=grid)
        elif 16 < M <= 32:
            mx_config = self.configurations['m32']
            block = (32, mx_config['block_row_warps'],
                     mx_config['block_col_warps'])
            MPAD = (M + mx_config['BM'] - 1) // mx_config['BM'] * mx_config['BM']
            grid = (N // mx_config['BN'], MPAD // mx_config['BM'], 1)
            self.m32(input, qweight, output, scales,
                     zeros, M, block=block, grid=grid)
        elif 32 < M <= 64:
            mx_config = self.configurations['m64']
            block = (32, mx_config['block_row_warps'],
                     mx_config['block_col_warps'])
            MPAD = (M + mx_config['BM'] - 1) // mx_config['BM'] * mx_config['BM']
            grid = (N // mx_config['BN'], MPAD // mx_config['BM'], 1)
            self.m64(input, qweight, output, scales,
                     zeros, M, block=block, grid=grid)
        elif 64 < M <= 128:
            mx_config = self.configurations['m128']
            block = (32, mx_config['block_row_warps'],
                     mx_config['block_col_warps'])
            MPAD = (M + mx_config['BM'] - 1) // mx_config['BM'] * mx_config['BM']
            grid = (N // mx_config['BN'], MPAD // mx_config['BM'], 1)
            self.m128(input, qweight, output, scales,
                      zeros, M, block=block, grid=grid)
        else:
            mx_config = self.configurations['m256']
            block = (32, mx_config['block_row_warps'],
                     mx_config['block_col_warps'])
            MPAD = (M + mx_config['BM'] - 1) // mx_config['BM'] * mx_config['BM']
            grid = (N // mx_config['BN'], MPAD // mx_config['BM'], 1)
            self.m256(input, qweight, output, scales,
                      zeros, M, block=block, grid=grid)

    def _get_executable_m1(self, bits: int, n: int, k: int, group_size: int = -1):
        # get src code
        m1_module = get_gemv_workloads(bits, n, k)
        num_warps = self.configurations['m1']['num_warps']
        m1_mod = self._apply_gemv_schedule(m1_module, bits, k, num_warps)
        code = m1_mod.imported_modules[0].get_source()
        name = f"tir_halfxint{bits}_simt_bn{num_warps}_k{k}"
        code = code.replace(
            "main_kernel0", name)
        code = code.split("extern \"C\"")[1]
        code = "extern \"C\"" + code
        code = "#include <cuda_fp16.h>\n" + code
        return TVMExecutable(code, name)

    def _get_executable_mx(self, bits: int, m:int, n: int, k: int, group_size: int = -1):
        mx = f'm{m}'
        mx_config = self.configurations[mx]
        print('generate', mx)
        mx_mod = _apply_dynamic_gemm_schedule(bits, m, n, k, group_size, mx_config)
        name = f"tir_halfxint{bits}_tensorop_{mx_config['BM']}x{mx_config['BN']}x{mx_config['BK']}x{mx_config['stage']}_t{mx_config['raster']}_y{mx_config['block_row_warps']}z{mx_config['block_col_warps']}_K{k}_align8"
        code = mx_mod.imported_modules[0].get_source()
        code = code.replace(
            "main_kernel0", name)
        code = code.split("extern \"C\"")[1]
        code = "extern \"C\"" + code
        code = '''
            static inline __device__ __host__ unsigned
            __pack_half2(const half x, const half y) {
            unsigned v0 = *((unsigned short *)&x);
            unsigned v1 = *((unsigned short *)&y);
            return (v1 << 16) | v0;
        }''' + code
        code = "#include <mma.h>\n" + code
        code = "#include <cuda_fp16.h>\n" + code
        return TVMExecutable(code, name)

    def _tune(self, n, k, bits):
        experiment = Experiment('local')
        experiment.config.experiment_working_directory = self.working_directory
        experiment.config.experiment_name = f'_autogptq_search_{experiment.id}'
        experiment.config.tuner.name = 'Evolution'
        experiment.config.tuner.class_args['optimize_mode'] = 'minimize'
        experiment.config.tuner.class_args['population_size'] = 2048
        experiment.config.max_trial_number = self.trails
        experiment.config.trial_concurrency = 32
        experiment.config.trial_gpu_number = 4
        experiment.config.tuner_gpu_indices = [0, 1, 2, 3]
        experiment.config.use_annotation = False
        experiment.config.training_service.use_active_gpu = True
        experiment.config.training_service.platform = 'local'
        experiment.config.training_service.trial_gpu_number = 4
        experiment.config.training_service.max_trial_number_per_gpu = 13
        experiment.config.training_service.gpu_indices = [0, 1, 2, 3]
        search_space = {
                "block_col_warps": {"_type": "choice", "_value": [1, 2, 4, 8, 16]},
                "BN": {"_type": "choice", "_value": [64, 128, 256]},
                "BK": {"_type": "choice", "_value": [16, 32, 64]},
                # since the matrix is small, we don't need raster
                "raster": {"_type": "choice", "_value": [0]},
                # async_copy is always enbaled becuase currently we use the pad with conflict implementation.
                "stage": {"_type": "choice", "_value": [1, 2, 3]},
        }
        for m in self.m_candidates:
            if m == 1:
                continue
            port = self.port_pool.get_port()
            experiment_id = f'gemm_{bits}bit_{m}x{n}x{k}'
            experiment.id = experiment_id
            search_space['block_row_warps'] = {"_type": "choice", "_value": [(pow(2, i))for i in range(0, 5) if (16 * pow(2, i)) <= m]}
            search_space['BM'] = {"_type": "choice", "_value": [(16 * pow(2, i))for i in range(0, 5) if ( 16 * pow(2, i) ) <= m]}
            experiment.config.search_space = search_space
            # get the wrokloads.py from current file path
            experiment.config.trial_command = f"python {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'workloads.py')} --M {m} --N {n} --K {k} --bits {bits}"

            if os.path.exists(os.path.join(self.working_directory, experiment_id, 'db/nni.sqlite')):
                # experiment.resume(experiment_id=experiment_id, port=port, wait_completion=True)
                # experiment.stop()
                ...
            else:
                experiment.run(port=port, wait_completion=True)         
                experiment.stop()
            time.sleep(5) # wait for the port to be released
            db_path = os.path.join(self.working_directory, experiment_id, 'db/nni.sqlite')
            db = NNIDatabase(db_path)
            db.connect()
            best_params = db.get_best_params()
            db.close()
            self.configurations[f'm{m}'] = best_params
        


def bit_compress(x, bits, axis):
    if bits == 3:
        # given a tensor x (M, K), which only the low bits bits have value, we can compress it to (M, K // 8 * bits)
        shape = x.shape
        qshape = shape[:axis] + (shape[axis] // 32 * bits,) + shape[axis + 1:]
        qweight = np.zeros(qshape).astype("int32")
        mask = (1 << bits) - 1
        for row in range(qweight.shape[0]):
            # print("compressing: ", row)
            weight = x[row]
            # compress consective 32 weight 32-bit(actually is only 3bit value) integers into 3 32-bit integers
            i = 0
            col = 0
            while col < qweight.shape[1]:
                for j in range(i, i + 10):
                    qweight[row, col] |= weight[j] << (3 * (j - i))
                i += 10
                qweight[row, col] |= weight[i] << 30
                col += 1
                qweight[row, col] |= (weight[i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):
                    qweight[row, col] |= weight[j] << (3 * (j - i) + 1)
                i += 10
                qweight[row, col] |= weight[i] << 31
                col += 1
                qweight[row, col] |= (weight[i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):
                    qweight[row, col] |= weight[j] << (3 * (j - i) + 2)
                i += 10
                col += 1
        # convert to int8 in meomory view
        qweight = qweight.view("int8")
        return qweight
    elif bits == 4:
        shape = x.shape
        compress_shape = shape[:axis] + \
            (shape[axis] // 8 * bits,) + shape[axis + 1:]
        _compressed = np.zeros(compress_shape).astype("int8")
        mask = (1 << bits) - 1
        for i in range(shape[axis]):
            val = (x[..., i] & mask).astype("int8")
            _compressed[..., i // (8 // bits)] |= (val <<
                                                   ((i % (8 // bits)) * bits)).astype("int8")
        return _compressed


def gemv_test(M, N, K):
    handler = TVMHandler(bits=3, n=N, k=K)
    x = torch.rand((M, K), dtype=torch.float16).cuda()
    w = (np.arange(N * K) % 4).reshape((N, K)).astype("int8")
    qw = bit_compress(w, 3, 1)
    print(np.matmul(x.cpu().numpy(), w.T))
    w = torch.from_numpy(w).cuda()
    qw = torch.from_numpy(qw).cuda()
    scales = torch.ones(N, dtype=torch.float16).cuda()
    zeros = torch.zeros(N, dtype=torch.float16).cuda()
    y = torch.zeros((M, N), dtype=torch.float16).cuda()
    handler(x, qw, y, scales, zeros)
    print(y.cpu().numpy())
    return y


if __name__ == '__main__':
    # test for gemv kernel
    M = 1
    N = 1024
    K = 768
    gemv_test(M, N, K)
    # test for 3x3 kernel
    # M = 16
    # N = 1024
    # K = 768
    # handler = TVMHandler(bits=3, n=N, k=K)
    # x = torch.ones((M, K), dtype=torch.float16).cuda()
    # w = (np.arange(N * K) % 4).reshape((N, K)).astype("int8")
    # qw = bit_compress(w, 3, 1)
    # print(np.matmul(x.cpu().numpy(), w.T))
    # w = torch.from_numpy(w).cuda()
    # qw = torch.from_numpy(qw).cuda()
    # scales = torch.ones(N, dtype=torch.float16).cuda()
    # zeros = torch.zeros(N, dtype=torch.float16).cuda()
    # y = torch.zeros((M, N), dtype=torch.float16).cuda()
    # handler(x, qw, y, scales, zeros)
    # print(y)
