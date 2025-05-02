import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for SELU activation
selu_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define ALPHA 1.6732632423540128f
#define SCALE 1.0507009873554805f

__global__ void selu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = (x > 0.0f) ? SCALE * x : SCALE * ALPHA * (exp(x) - 1.0f);
    }
}

torch::Tensor selu_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    selu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

selu_cpp_source = "torch::Tensor selu_cuda(torch::Tensor input);"

# Compile the inline CUDA code for SELU
selu_op = load_inline(
    name="selu_cuda",
    cpp_sources=selu_cpp_source,
    cuda_sources=selu_cuda_source,
    functions=["selu_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA kernel for SELU activation.
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        self.selu_op = selu_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies SELU activation to the input tensor using a custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with SELU applied, same shape as input.
        """
        return self.selu_op.selu_cuda(x)
