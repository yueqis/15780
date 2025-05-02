import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for mean reduction along a specific dimension
mean_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void mean_reduction_kernel(const scalar_t* input, scalar_t* output, int input_size, int reduce_dim_size, int outer_dims, int inner_dims) {
    extern __shared__ float sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate the start index in the input tensor for this thread's computation
    int outer_idx = idx / (inner_dims);
    int inner_idx = idx % inner_dims;

    const scalar_t* x = input + outer_idx * reduce_dim_size * inner_dims + inner_idx;
    scalar_t sum = 0;

    // Perform the reduction over the specified dimension
    for (int i = 0; i < reduce_dim_size; ++i) {
        sum += x[i * inner_dims];
    }

    // Store intermediate results in shared memory
    sdata[threadIdx.x] = sum;
    __syncthreads();

    // Finalize the mean by dividing by the reduced dimension size
    if (threadIdx.x == 0) {
        sum = 0;
        for (int i = 0; i < blockDim.x; ++i) {
            sum += sdata[i];
        }
        output[outer_idx * inner_dims + inner_idx] = sum / reduce_dim_size;
    }
}

torch::Tensor mean_reduction_cuda(torch::Tensor input, int64_t dim) {
    // Handle negative dimensions (PyTorch-style)
    if (dim < 0) {
        dim += input.dim();
    }

    // Ensure that the dimension is valid
    TORCH_CHECK(dim >= 0 && dim < input.dim(), "Invalid reduction dimension");

    // Permute dimensions so that the reduction dimension is at index -1
    std::vector<int64_t> dims_order;
    for (int64_t i = 0; i < input.dim(); ++i) {
        if (i != dim) {
            dims_order.push_back(i);
        }
    }
    dims_order.push_back(dim);
    input = input.permute(dims_order);

    // Flatten all except the last dimension
    int outer_dims = 1;
    for (int i = 0; i < input.dim() - 1; ++i) {
        outer_dims *= input.size(i);
    }
    int inner_dims = input.size(-2);
    int reduce_dim_size = input.size(-1);

    auto output = torch::zeros({outer_dims, inner_dims}, input.options());

    const int threads = 256;
    const int blocks = (outer_dims * inner_dims + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mean_reduction_cuda", ([&] {
        mean_reduction_kernel<scalar_t><<<blocks, threads, threads * sizeof(scalar_t)>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input.numel(),
            reduce_dim_size,
            outer_dims,
            inner_dims
        );
    }));

    // Reshape output to match PyTorch's torch.mean output shape
    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < input.dim() - 1; ++i) {
        output_shape.push_back(input.size(i));
    }

    return output.view(output_shape);
}
"""

mean_reduction_cpp_source = """
torch::Tensor mean_reduction_cuda(torch::Tensor input, int64_t dim);
"""

# Compile the inline CUDA code for mean reduction
mean_reduction_op = load_inline(
    name="mean_reduction",
    cpp_sources=mean_reduction_cpp_source,
    cuda_sources=mean_reduction_source,
    functions=["mean_reduction_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.mean_reduction = mean_reduction_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mean_reduction.mean_reduction_cuda(x, self.dim)


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs mean reduction over a specific dimension.
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
        Reduces the input tensor along the specified dimension by taking the mean.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with reduced dimension. The shape of the output is the same as the input except for the reduced dimension which is removed.
        """
        return torch.mean(x, dim=self.dim)

batch_size = 16
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return [1]
