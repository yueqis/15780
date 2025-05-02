import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Softplus
softplus_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ float softplus_device(float x) {
    // Softplus: log(1 + exp(x))
    return logf(1.0f + expf(x));
}

__global__ void softplus_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = softplus_device(input[idx]);
    }
}

torch::Tensor softplus_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    softplus_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    return output;
}
"""

softplus_cpp_source = "torch::Tensor softplus_cuda(torch::Tensor input);"

# Compile the inline CUDA code
softplus_op = load_inline(
    name="softplus_op",
    cpp_sources=softplus_cpp_source,
    cuda_sources=softplus_cuda_source,
    functions=["softplus_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softplus_op = softplus_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softplus_op.softplus_cuda(x)
