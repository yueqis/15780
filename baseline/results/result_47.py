import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Inline CUDA kernel for sum reduction along a specified dimension
sum_reduce_cuda_code = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void sum_reduce_kernel(const scalar_t* input, scalar_t* output, 
                                  int dim_size, int reduce_dim, 
                                  int outer_dims, int inner_dims) {
    const int outer = blockIdx.x / inner_dims;
    const int inner = blockIdx.x % inner_dims;
    const int tid = threadIdx.x;

    const scalar_t* input_slice = input + (outer * reduce_dim + tid) * inner_dims + inner;
    scalar_t sum = 0;

    for (int i = 0; i < reduce_dim; ++i) {
        sum += input_slice[i * inner_dims];
    }

    output[outer * inner_dims + inner] = sum;
}

torch::Tensor sum_reduce_cuda(torch::Tensor input, int reduce_dim) {
    int64_t outer_dims = 1;
    for (int i = 0; i < reduce_dim; ++i) {
        outer_dims *= input.size(i);
    }

    int64_t inner_dims = 1;
    for (int i = reduce_dim + 1; i < input.dim(); ++i) {
        inner_dims *= input.size(i);
    }

    int64_t dim_size = input.size(reduce_dim);

    auto output = torch::zeros({input.size(0), inner_dims}, input.options());

    const int threads_per_block = dim_size;
    const int blocks_per_grid = outer_dims * inner_dims;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "sum_reduce_cuda", ([&] {
        sum_reduce_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim_size,
            reduce_dim,
            outer_dims,
            inner_dims
        );
    }));

    // Reshape to match the original dimensions with keepdim=True
    std::vector<int64_t> shape;
    for (int i = 0; i < input.dim(); ++i) {
        if (i == reduce_dim)
            shape.push_back(1);
        else
            shape.push_back(-1);
    }

    return output.view(shape);
}
"""

sum_reduce_cpp_decl = """
torch::Tensor sum_reduce_cuda(torch::Tensor input, int reduce_dim);
"""

# Compile the inline CUDA code
sum_reduce_op = load_inline(
    name="sum_reduce",
    cpp_sources=sum_reduce_cpp_decl,
    cuda_sources=sum_reduce_cuda_code,
    functions=["sum_reduce_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA kernel for sum reduction.
    """

    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.sum_reduce = sum_reduce_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sum_reduce.sum_reduce_cuda(x, self.dim)
