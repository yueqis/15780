import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for masked cumulative sum
masked_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void masked_cumsum_kernel(const float* x, const bool* mask, float* out, int size, int dim_size, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int batch_idx = idx / dim_size;
        int dim_idx = idx % dim_size;
        float sum = 0.0f;
        for (int i = 0; i <= dim_idx; ++i) {
            int pos = batch_idx * dim_size + i;
            if (mask[pos]) {
                sum += x[pos];
            }
        }
        out[idx] = sum;
    }
}

torch::Tensor masked_cumsum_cuda(torch::Tensor x, torch::Tensor mask, int dim) {
    auto sizes = x.sizes();
    auto out = torch::zeros_like(x);
    int dim_size = sizes[dim];
    int total_size = 1;
    for (int i = 0; i < x.dim(); ++i) {
        if (i != dim) {
            total_size *= sizes[i];
        }
    }
    int stride = 1;
    for (int i = dim + 1; i < x.dim(); ++i) {
        stride *= sizes[i];
    }

    const int block_size = 256;
    const int num_blocks = (x.numel() + block_size - 1) / block_size;
    masked_cumsum_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), mask.data_ptr<bool>(), out.data_ptr<float>(), x.numel(), dim_size, stride);
    return out;
}
"""

masked_cumsum_cpp_source = (
    "torch::Tensor masked_cumsum_cuda(torch::Tensor x, torch::Tensor mask, int dim);"
)

# Compile the inline CUDA code for masked cumulative sum
masked_cumsum_op = load_inline(
    name="masked_cumsum",
    cpp_sources=masked_cumsum_cpp_source,
    cuda_sources=masked_cumsum_source,
    functions=["masked_cumsum_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.masked_cumsum = masked_cumsum_op

    def forward(self, x, mask):
        return self.masked_cumsum.masked_cumsum_cuda(x, mask, self.dim)


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs a masked cumulative sum, only summing elements that satisfy a condition.

    Parameters:
        dim (int): The dimension along which to perform the masked cumulative sum.
    """

    def __init__(self, dim):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x, mask):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).
            mask (torch.Tensor): Boolean mask of the same shape as x.

        Returns:
            torch.Tensor: Cumulative sum of elements where mask is True.
        """
        return torch.cumsum(x * mask, dim=self.dim)

batch_size = 128
input_shape = (4000,)
dim = 1

def get_inputs():
    x = torch.randn(batch_size, *input_shape)
    mask = torch.randint(0, 2, x.shape).bool()  # Random boolean mask
    return [x, mask]

def get_init_inputs():
    return [dim]

