import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Inline CUDA kernel for masked cumsum
masked_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Custom CUDA kernel to compute masked cumulative sum along dim=1
__global__ void masked_cumsum_kernel(const float* x, const bool* mask, float* out, int rows, int cols) {
    // Each thread block handles one row
    int row = blockIdx.x;
    int col_start = threadIdx.x;
    int stride = blockDim.x;

    int idx = row * cols + col_start;
    float running_sum = 0.0f;

    for (int col = col_start; col < cols; col += stride) {
        idx = row * cols + col;
        if (mask[idx]) {
            running_sum += x[idx];
        }
        out[idx] = running_sum;
    }
}

torch::Tensor masked_cumsum_cuda(torch::Tensor x, torch::Tensor mask) {
    auto size = x.sizes();
    int batch_size = size[0];
    int feature_dim = size[1];

    auto out = torch::zeros_like(x);

    dim3 grid(batch_size);
    dim3 block(256);

    masked_cumsum_kernel<<<grid, block>>>(x.data_ptr<float>(), mask.data_ptr<bool>(), out.data_ptr<float>(), batch_size, feature_dim);

    return out;
}
"""

masked_cumsum_cpp_source = """
torch::Tensor masked_cumsum_cuda(torch::Tensor x, torch::Tensor mask);
"""

# Compile the inline CUDA code
masked_cumsum_op = load_inline(
    name="masked_cumsum",
    cpp_sources=masked_cumsum_cpp_source,
    cuda_sources=masked_cumsum_source,
    functions=["masked_cumsum_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernel for masked cumulative sum.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.masked_cumsum = masked_cumsum_op

    def forward(self, x, mask):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).
            mask (torch.Tensor): Boolean mask of the same shape as x.

        Returns:
            torch.Tensor: Cumulative sum of elements where mask is True.
        """
        # We only support dim=1 in our custom kernel
        if self.dim != 1:
            raise ValueError("Custom masked cumsum kernel only supports dim=1")

        return self.masked_cumsum.masked_cumsum_cuda(x, mask)


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

