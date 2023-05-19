import math
import numpy as np
import torch
import torch.nn as nn
from .tvm_untils import cache

is_mlc_llm=False
# Assumes layer is perfectly divisible into 1024 * 1024 blocks
class QuantLinear(nn.Module): 

    def __init__(
        self,
        bits,
        groupsize,
        infeatures,
        outfeatures,
        bias,
        ):
        super().__init__()
        if bits not in [2, 3, 4, 8]:
            raise NotImplementedError("Only 2,3,4,8 bits are supported.")
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.groupsize = groupsize if groupsize != -1 else infeatures
        self.register_buffer(
            'qweight', torch.zeros((outfeatures, infeatures // 8 * 3), dtype=torch.int8 if not is_mlc_llm else torch.int32)
        )
        self.register_buffer('scales', torch.zeros((math.ceil(infeatures / self.groupsize), outfeatures), dtype=torch.float16))
        self.register_buffer('zeros', torch.zeros((math.ceil(infeatures / self.groupsize), outfeatures), dtype=torch.float16))
        if bias:
            self.register_buffer('bias', torch.zeros((outfeatures), dtype=torch.float16))
        else:
            self.bias = None
        self.groupsize = groupsize if groupsize != -1 else infeatures

        if groupsize != -1:
            self.register_buffer(
                'g_idx',
                torch.tensor([i // self.groupsize for i in range(infeatures)], dtype=torch.int32)
            )
        self.tvm_handler = cache.get_handler(n=outfeatures, k=infeatures, bits=bits, group_size=groupsize)
    
    def pack(self, linear, scales, zeros, g_idx=None):
        W = linear.weight.data.clone()

        self.g_idx = g_idx.clone() if g_idx is not None else self.g_idx
        print("scales shape", scales.shape, zeros.shape, self.g_idx.shape)
        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous()
        scale_zeros = zeros * scales
        self.scales = scales.clone().half()
        self.zeros = scale_zeros.clone().half()
        if linear.bias is not None:
            self.bias = linear.bias.clone().half()

        intweight = []
        for idx in range(self.infeatures):
            intweight.append(
                torch.round(
                    (
                        W[:, idx] + scale_zeros[self.g_idx[idx]]) / self.scales[self.g_idx[idx]]
                ).to(torch.int)[:, None]
            )
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)

        i = 0
        row = 0
        qweight = np.zeros(
            (intweight.shape[0] // 32 * self.bits, intweight.shape[1]), dtype=np.uint32
        )
        while row < qweight.shape[0]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qweight[row] |= intweight[j] << (self.bits * (j - i))
                i += 32 // self.bits
                row += 1
            elif self.bits == 3:
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i))
                i += 10
                qweight[row] |= intweight[i] << 30
                row += 1
                qweight[row] |= (intweight[i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 1)
                i += 10
                qweight[row] |= intweight[i] << 31
                row += 1
                qweight[row] |= (intweight[i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 2)
                i += 10
                row += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")
        if is_mlc_llm:
            self.qweight = torch.from_numpy(qweight.astype(dtype=np.int32)) 
            zeros = -zeros
            self.zeros = -self.zeros
            return
        qweight = np.ascontiguousarray(qweight.T)
        qweight = qweight.view(dtype=np.int8)
        self.qweight = torch.from_numpy(qweight) 


    def forward(self, x):
        print('QuantLinear forward, xshape is ', x.shape)
        # print(x)
        dtype = x.dtype
        x = x.half()
        M = 1
        for i in range(len(x.shape) - 1):
            M *= x.shape[i]
        x = x.reshape((M, -1))
        outshape = x.shape[:-1] + (self.outfeatures,)
        pad = 0
        if x.shape[-1] == x.numel():
            y = torch.zeros(outshape, dtype=x.dtype, device=x.device)
            self.tvm_handler(x, self.qweight, y, self.scales, self.zeros)
            y = y.reshape(outshape)
            y = y + self.bias if self.bias is not None else y 
            return y 
        elif 1 < M <= 16:
            if M % 16 != 0:
                pad = 16 - x.shape[0] % 16
        elif 16 < M <= 32:
            if x.shape[0] % 32 != 0:
                pad = 32 - x.shape[0] % 32
        elif 32 < M <= 64:
            if x.shape[0] % 64 != 0:
                pad = 64 - x.shape[0] % 64
        elif 64 < M <= 128:
            if x.shape[0] % 128 != 0:
                pad = 128 - x.shape[0] % 128
        else:
            if x.shape[0] % 256 != 0:
                pad = 256 - x.shape[0] % 256
        x = torch.nn.functional.pad(x, (0, 0, 0, pad))
        y_pad = torch.zeros((outshape[0] + pad, outshape[-1]), dtype=x.dtype, device=x.device)
        # print(x.shape, outshape, pad, y_pad.shape)
        # print('x ', x)
        # print('y_pad ', y_pad)
        # print('qweight ', self.qweight)
        # print('scales ', self.scales)
        # print('zeros ', self.zeros)
        self.tvm_handler(x, self.qweight, y_pad, self.scales, self.zeros)
        # recover y_pad to y
        y = torch.zeros(outshape, dtype=dtype, device=x.device)
        y[:M] = y_pad[:M]
        y = y + self.bias if self.bias is not None else y 
        y.to(dtype)
        print(y)
        # print(y.shape)
        return y

        