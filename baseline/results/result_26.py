import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for GELU activation
gelu_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// CUDA kernel to compute GELU for each element
__global__ void gelu_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float xi = x[idx];
        // Approximate GELU using tanh approximation: x * 0.5 * (1.0 + tanh(sqrt(2/M_PI) * (x + 0.044715 * x^3)))
        float cdf = 0.5f * (1.0f + tanhf((0.7978845608f * (xi + 0.044715f * xi * xi * xi))));
        out[idx] = xi * cdf;
    }
}

// CUDA C++ interface
torch::Tensor gelu_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    gelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

gelu_cpp_declaration = "torch::Tensor gelu_cuda(torch::Tensor x);"

# Compile the inline CUDA code
gelu_op = load_inline(
    name="gelu_op",
    cpp_sources=gelu_cpp_declaration,
    cuda_sources=gelu_kernel_source,
    functions=["gelu_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA implementation of GELU activation.
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        self.gelu_op = gelu_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies GELU activation using a custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with GELU applied, same shape as input.
        """
        return self.gelu_op.gelu_cuda(x)


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a GELU activation.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies GELU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with GELU applied, same shape as input.
        """
        return torch.nn.functional.gelu(x)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed
