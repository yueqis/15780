import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Inline CUDA kernel for min reduction along a specific dimension
min_reduce_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void min_reduce_kernel(
    const scalar_t* input,
    scalar_t* output,
    int dim_size,
    int reduce_dim_stride,
    int outer_dims) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer_dims) return;

    const scalar_t* in_ptr = input + idx * reduce_dim_stride;
    scalar_t min_val = in_ptr[0];

    for (int i = 1; i < dim_size; ++i) {
        min_val = fminf(min_val, in_ptr[i]);
    }

    output[idx] = min_val;
}

torch::Tensor min_reduce_cuda(torch::Tensor input, int dim) {
    int64_t ndim = input.dim();
    if (dim < 0) {
        dim += ndim;
    }

    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid reduction dimension");

    int64_t dim_size = input.size(dim);
    int64_t total_elems = input.numel();
    int64_t reduce_dim_stride = input.stride(dim);

    // Collapse outer dimensions
    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;
    for (int64_t d = 0; d < ndim; ++d) {
        if (d != dim) {
            sizes.push_back(input.size(d));
            strides.push_back(input.stride(d));
        }
    }
    int64_t outer_dims = input.size(0) * input.size(1) * input.size(2) / dim_size;

    auto output = torch::empty(sizes, input.options());

    const int blockSize = 256;
    const int numBlocks = (outer_dims + blockSize - 1) / blockSize;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "min_reduce_cuda", ([&] {
        min_reduce_kernel<scalar_t><<<numBlocks, blockSize>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim_size,
            reduce_dim_stride,
            outer_dims);
    }));

    return output;
}
"""

min_reduce_cpp_source = """
torch::Tensor min_reduce_cuda(torch::Tensor input, int dim);
"""

# Compile the inline CUDA extension
min_reduce_extension = load_inline(
    name="min_reduce_extension",
    cpp_sources=min_reduce_cpp_source,
    cuda_sources=min_reduce_cuda_source,
    functions=["min_reduce_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA kernel for min reduction.
    """

    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.min_reduce_cuda = min_reduce_extension.min_reduce_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.min_reduce_cuda(x, self.dim)


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs min reduction over a specific dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies min reduction over the specified dimension to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after min reduction over the specified dimension.
        """
        return torch.min(x, dim=self.dim)[0]

batch_size = 16
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return [1] # Example, change to desired dimension
