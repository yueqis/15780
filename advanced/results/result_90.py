import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for cumulative product along a dimension
cumprod_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void cumprod_kernel(const float* input, float* output, int size, int dim_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int stride = 1;
        int offset = idx;
        for (int i = 0; i < dim; ++i) {
            stride *= input_shape[i];
        }
        int outer_idx = offset / (dim_size * stride);
        int inner_idx = (offset % (dim_size * stride)) / stride;
        int base_idx = outer_idx * dim_size * stride + inner_idx * stride;

        float running_prod = 1.0f;
        for (int i = 0; i < dim_size; ++i) {
            int read_idx = base_idx + i * stride;
            running_prod *= input[read_idx];
            output[read_idx] = running_prod;
        }
    }
}

torch::Tensor cumprod_cuda(torch::Tensor input, int dim) {
    auto input_sizes = input.sizes();
    auto output = torch::zeros_like(input);
    int64_t total_size = 1;
    for (auto s : input_sizes) total_size *= s;
    
    int dim_size = input_sizes[dim];
    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;
    
    cumprod_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), total_size, dim_size, dim, input_sizes);
    return output;
}
"""

cumprod_cpp_source = """
torch::Tensor cumprod_cuda(torch::Tensor input, int dim);
"""

# Compile the inline CUDA code
cumprod_op = load_inline(
    name="cumprod_op",
    cpp_sources=cumprod_cpp_source,
    cuda_sources=cumprod_cuda_source,
    functions=["cumprod_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA kernel to perform cumulative product along a specified dimension.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.cumprod_op = cumprod_op

    def forward(self, x):
        return self.cumprod_op.cumprod_cuda(x, self.dim)
