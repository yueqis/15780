import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for cumulative product along dim=1
cumprod_dim1_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel to compute cumulative product across columns (dim=1 assumed)
__global__ void cumprod_dim1_kernel(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x;
    if (row < rows) {
        int offset = row * cols;
        float running_prod = 1.0f;
        for (int col = 0; col < cols; ++col) {
            int idx = offset + col;
            running_prod *= input[idx];
            output[idx] = running_prod;
        }
    }
}

torch::Tensor cumprod_dim1_cuda(torch::Tensor input) {
    int64_t rows = input.size(0);
    int64_t cols = input.numel() / rows;

    auto output = torch::empty_like(input);

    cumprod_dim1_kernel<<<rows, 1>>>(input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);

    return output;
}
"""

cumprod_dim1_cpp_source = "torch::Tensor cumprod_dim1_cuda(torch::Tensor input);"

# Compile the inline CUDA code
cumprod_dim1_op = load_inline(
    name="cumprod_dim1",
    cpp_sources=cumprod_dim1_cpp_source,
    cuda_sources=cumprod_dim1_cuda_source,
    functions=["cumprod_dim1_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA kernel for cumulative product along dim=1.
    Assumes that the dimension to operate on is known at compile time and fixed.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.cumprod_op = cumprod_dim1_op

    def forward(self, x):
        # Only support dim=1 in this optimized version
        if self.dim != 1:
            raise ValueError(
                "This optimized model only supports cumulative product along dim=1"
            )
        return self.cumprod_op.cumprod_dim1_cuda(x)


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs a cumulative product operation along a specified dimension.

    Parameters:
        dim (int): The dimension along which to perform the cumulative product operation.
    """

    def __init__(self, dim):
        """
        Initialize the CumulativeProductModel.

        Args:
            dim (int): The dimension along which to perform the cumulative product.
        """
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x):
        """
        Forward pass, computing the cumulative product along the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).

        Returns:
            torch.Tensor: Tensor of the same shape as `x` after applying cumulative product along `dim`.
        """
        return torch.cumprod(x, dim=self.dim)

# Define input dimensions and parameters
batch_size = 128
input_shape = (4000,)
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return [dim]

