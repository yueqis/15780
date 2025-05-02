import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for product reduction along a dimension
product_reduce_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

// Custom CUDA kernel to compute product along a specific dimension
template <typename scalar_t>
__global__ void product_reduce_kernel(const scalar_t* input, scalar_t* output,
                                      int outer_size, int inner_size, int reduce_size) {
    // blockIdx.x determines which outer block we are in
    // blockIdx.y determines which inner feature we are computing
    int batch_idx = blockIdx.x;
    int inner_idx = blockIdx.y;

    // Each thread will handle a portion of the reduction dimension
    scalar_t result = 1.0;

    for (int i = threadIdx.x; i < reduce_size; i += blockDim.x) {
        int input_index = batch_idx * inner_size * reduce_size + inner_idx * reduce_size + i;
        result *= input[input_index];
    }

    // Atomic multiply to accumulate the result in shared memory
    __shared__ scalar_t shared_result[THREADS_PER_BLOCK];
    shared_result[threadIdx.x] = result;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_result[threadIdx.x] *= shared_result[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[batch_idx * inner_size + inner_idx] = shared_result[0];
    }
}

torch::Tensor product_reduce_cuda(torch::Tensor input, int dim) {
    // Ensure input is on GPU and contiguous
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    // Get input dimensions
    auto sizes = input.sizes();
    int ndims = sizes.size();
    
    // Normalize the dimension
    if (dim < 0) dim += ndims;
    TORCH_CHECK(dim >= 0 && dim < ndims, "Invalid reduction dimension");

    // Handle empty tensors
    if (input.numel() == 0) {
        return torch::empty_like(input, {sizes[0], sizes[1], sizes[2]}, at::MemoryFormat::Contiguous);
    }

    // Determine outer size, inner size, and reduce size based on the reduction dimension
    int outer_size = 1;
    for (int i = 0; i < dim; ++i) {
        outer_size *= sizes[i];
    }

    int reduce_size = sizes[dim];

    int inner_size = 1;
    for (int i = dim + 1; i < ndims; ++i) {
        inner_size *= sizes[i];
    }

    // Prepare output tensor
    std::vector<int64_t> out_sizes;
    for (int i = 0; i < ndims; ++i) {
        if (i != dim) out_sizes.push_back(sizes[i]);
    }
    auto output = torch::zeros(torch::IntList(out_sizes), input.options());

    // Launch kernel
    dim3 grid(outer_size, inner_size);
    dim3 block(THREADS_PER_BLOCK);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "product_reduce_cuda", ([&] {
        product_reduce_kernel<scalar_t><<<grid, block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer_size,
            inner_size,
            reduce_size
        );
    }));

    return output;
}
"""

product_reduce_cpp_source = """
torch::Tensor product_reduce_cuda(torch::Tensor input, int dim);
"""

# Compile the inline CUDA code
product_reduce_op = load_inline(
    name="product_reduce",
    cpp_sources=product_reduce_cpp_source,
    cuda_sources=product_reduce_source,
    functions=["product_reduce_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.product_reduce = product_reduce_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.product_reduce.product_reduce_cuda(x, self.dim)


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
