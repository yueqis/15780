import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Inline CUDA kernel for product reduction
prod_reduce_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void prod_reduce_kernel(const scalar_t* input, scalar_t* output, 
                                   int dim_size, int reduce_dim, int outer_dims) {
    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= outer_dims) return;

    // Calculate the input index without the reduction dimension
    int64_t before_reduce = tid / (dim_size / blockDim.y);
    int64_t after_reduce = tid % (dim_size / blockDim.y);

    scalar_t result = 1.0;
    for (int i = 0; i < reduce_dim; ++i) {
        int64_t idx = before_reduce * dim_size + i * (dim_size / reduce_dim) + after_reduce;
        result *= input[idx];
    }

    output[tid] = result;
}

torch::Tensor prod_reduce_cuda(torch::Tensor input, int64_t dim) {
    int64_t input_dims = input.dim();
    int64_t reduce_dim = input.size(dim);
    std::vector<int64_t> output_sizes;
    for (int64_t i = 0; i < input_dims; ++i) {
        if (i != dim) {
            output_sizes.push_back(input.size(i));
        }
    }

    torch::Tensor output = torch::zeros(output_sizes, input.options());

    const int64_t outer_dims = input.numel() / reduce_dim;
    const int64_t threads_per_block = 256;
    const int64_t blocks = (outer_dims + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "prod_reduce_cuda", ([&] {
        prod_reduce_kernel<scalar_t><<<blocks, dim3(threads_per_block, reduce_dim)>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input.size(dim),
            reduce_dim,
            outer_dims
        );
    }));

    return output;
}
"""

prod_reduce_cpp_source = """
torch::Tensor prod_reduce_cuda(torch::Tensor input, int64_t dim);
"""

# Compile the inline CUDA code
prod_reduce_extension = load_inline(
    name="prod_reduce",
    cpp_sources=prod_reduce_cpp_source,
    cuda_sources=prod_reduce_source,
    functions=["prod_reduce_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA kernel for product reduction.
    """

    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.prod_reduce = prod_reduce_extension.prod_reduce_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prod_reduce(x, self.dim)


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs product reduction over a dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
        """
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs product reduction over the specified dimension.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with product reduction applied.
        """
        return torch.prod(x, dim=self.dim)

batch_size = 16
dim1 = 256
dim2 = 256
reduction_dim = 1

def get_inputs():
    x = torch.randn(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return [reduction_dim]
