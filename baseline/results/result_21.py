import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Sigmoid activation
sigmoid_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void sigmoid_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

torch::Tensor sigmoid_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    sigmoid_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

sigmoid_cpp_source = "torch::Tensor sigmoid_cuda(torch::Tensor input);"

# Compile the inline CUDA code for Sigmoid
sigmoid_op = load_inline(
    name="sigmoid_cuda",
    cpp_sources=sigmoid_cpp_source,
    cuda_sources=sigmoid_cuda_source,
    functions=["sigmoid_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA implementation of Sigmoid activation.
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        self.sigmoid_op = sigmoid_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Sigmoid activation using a custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Sigmoid applied, same shape as input.
        """
        return self.sigmoid_op.sigmoid_cuda(x)


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a Sigmoid activation.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Sigmoid activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Sigmoid applied, same shape as input.
        """
        return torch.sigmoid(x)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed
