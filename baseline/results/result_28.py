import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for HardSigmoid activation
hardsigmoid_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hardsigmoid_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float clipped = fmaxf(fminf(x[idx] + 3.0f, 6.0f), 0.0f);
        out[idx] = clipped / 6.0f;
    }
}

torch::Tensor hardsigmoid_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    hardsigmoid_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

hardsigmoid_cpp_source = "torch::Tensor hardsigmoid_cuda(torch::Tensor x);"

# Compile the inline CUDA code for HardSigmoid
hardsigmoid_op = load_inline(
    name="hardsigmoid",
    cpp_sources=hardsigmoid_cpp_source,
    cuda_sources=hardsigmoid_cuda_source,
    functions=["hardsigmoid_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA kernel for HardSigmoid activation.
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        self.hardsigmoid_op = hardsigmoid_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies HardSigmoid activation using a custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with HardSigmoid applied, same shape as input.
        """
        return self.hardsigmoid_op.hardsigmoid_cuda(x)
