import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for min reduction along a specific dimension
min_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void min_reduction_kernel(
    const scalar_t* input,
    scalar_t* output,
    int input_dim0,
    int input_dim1,
    int input_dim2,
    int reduce_dim) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= input_dim0 * input_dim1)
        return;

    int batch_idx = idx / input_dim1;
    int channel_idx = idx % input_dim1;

    scalar_t min_val = input[batch_idx * input_dim1 * input_dim2 + channel_idx * input_dim2];

    for (int i = 1; i < input_dim2; ++i) {
        scalar_t val = input[batch_idx * input_dim1 * input_dim2 + channel_idx * input_dim2 + i];
        min_val = fminf(min_val, val);
    }

    output[batch_idx * input_dim1 + channel_idx] = min_val;
}

torch::Tensor min_reduction_cuda(torch::Tensor input, int reduce_dim) {
    int input_dim0 = input.size(0);
    int input_dim1 = input.size(1);
    int input_dim2 = input.size(2);

    auto output = torch::empty({input_dim0, input_dim1}, input.options());

    dim3 block(256);
    dim3 grid((input_dim0 * input_dim1 + block.x - 1) / block.x);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "min_reduction_cuda", ([&] {
        min_reduction_kernel<scalar_t><<<grid, block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input_dim0,
            input_dim1,
            input_dim2,
            reduce_dim);
    }));

    return output;
}
"""

min_reduction_cpp_source = """
torch::Tensor min_reduction_cuda(torch::Tensor input, int reduce_dim);
"""

# Compile the inline CUDA code
min_reduction_op = load_inline(
    name="min_reduction",
    cpp_sources=min_reduction_cpp_source,
    cuda_sources=min_reduction_source,
    functions=["min_reduction_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.min_reduction = min_reduction_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.min_reduction.min_reduction_cuda(x, self.dim)
