import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused GELU activation
gelu_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void gelu_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float xi = x[idx];
        float sqrt_2_over_pi = 0.7978845608f; // approx sqrt(2/pi)
        float coeff = 0.044715f;
        float inner = sqrt_2_over_pi * (xi + coeff * xi * xi * xi);
        float tanh_val = tanhf(inner);
        out[idx] = 0.5f * xi * (1.0f + tanh_val);
    }
}

torch::Tensor gelu_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    gelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

gelu_cpp_source = "torch::Tensor gelu_cuda(torch::Tensor x);"

# Compile the inline CUDA code for GELU
gelu_op = load_inline(
    name="gelu_op",
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_kernel_source,
    functions=["gelu_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.gelu_op = gelu_op

    def forward(self, x):
        return self.gelu_op.gelu_cuda(x)


batch_size = 2000
dim = 2000


def get_inputs():
    return [torch.randn(batch_size, dim).cuda()]


def get_init_inputs():
    return []


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# From https://github.com/karpathy/minGPT/blob/master/mingpt/model.py

class Model(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

batch_size = 2000
dim = 2000

def get_inputs():
    return [torch.randn(batch_size, dim)]

def get_init_inputs():
    return []
