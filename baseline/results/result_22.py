import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Tanh activation
tanh_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void tanh_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        // Compute tanh using its definition: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        float exp_pos = expf(x);
        float exp_neg = expf(-x);
        output[idx] = (exp_pos - exp_neg) / (exp_pos + exp_neg);
    }
}

torch::Tensor tanh_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    tanh_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

tanh_cpp_source = "torch::Tensor tanh_cuda(torch::Tensor input);"

# Compile the inline CUDA code for Tanh
tanh_op = load_inline(
    name="tanh_custom",
    cpp_sources=tanh_cpp_source,
    cuda_sources=tanh_kernel_source,
    functions=["tanh_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model that uses a custom CUDA kernel to compute Tanh activation.
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        self.tanh_op = tanh_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Tanh activation using the custom CUDA implementation.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Tanh applied, same shape as input.
        """
        return self.tanh_op.tanh_cuda(x)


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a Tanh activation.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Tanh activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Tanh applied, same shape as input.
        """
        return torch.tanh(x)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed
