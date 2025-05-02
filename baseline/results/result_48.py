import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Inline CUDA kernel for mean reduction along a specific dimension
mean_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void mean_kernel(const scalar_t* input, scalar_t* output, 
                            int dim_size, int reduce_size, int trailing_size) {
    extern __shared__ float sdata[];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    int num_per_output = reduce_size * trailing_size;
    int out_idx = i / num_per_output;
    int inner_idx = i % trailing_size;

    if (i < num_per_output * dim_size) {
        const scalar_t* in_slice = input + (out_idx * reduce_size * trailing_size);
        scalar_t sum = 0;
        for (int j = 0; j < reduce_size; ++j) {
            sum += in_slice[j * trailing_size + inner_idx];
        }

        // Use shared memory to reduce within block
        sdata[threadIdx.x] = sum;
        __syncthreads();

        // Block-level reduction
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sdata[threadIdx.x] += sdata[threadIdx.x + s];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            output[out_idx * trailing_size + inner_idx] = sum / reduce_size;
        }
    }
}

torch::Tensor mean_cuda(torch::Tensor input, int dim) {
    int input_dim = input.dim();
    if (dim < 0) {
        dim += input_dim;
    }

    // Handle negative dimensions and out-of-bound cases
    TORCH_CHECK(dim >= 0 && dim < input_dim, "Invalid dimension for mean reduction");

    // Get sizes
    int64_t dim_size = input.size(dim);
    int64_t trailing_size = 1;
    for (int i = dim + 1; i < input_dim; ++i) {
        trailing_size *= input.size(i);
    }

    int64_t leading_size = 1;
    for (int i = 0; i < dim; ++i) {
        leading_size *= input.size(i);
    }

    int64_t reduce_size = input.size(dim);

    // Create output tensor
    std::vector<int64_t> output_sizes;
    for (int i = 0; i < input_dim; ++i) {
        if (i != dim) {
            output_sizes.push_back(input.size(i));
        }
    }

    torch::Tensor output = torch::empty(output_sizes, input.options());

    const int threads = 256;
    const int blocks = (leading_size * trailing_size + threads - 1) / threads;

    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(input.type(), "mean_cuda", ([&] {
        mean_kernel<scalar_t><<<blocks, threads, threads * sizeof(scalar_t)>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim_size,
            reduce_size,
            trailing_size
        );
    }));

    return output;
}
"""

mean_cpp_source = """
torch::Tensor mean_cuda(torch::Tensor input, int dim);
"""

# Compile the inline CUDA code
mean_op = load_inline(
    name="mean_op",
    cpp_sources=mean_cpp_source,
    cuda_sources=mean_reduction_source,
    functions=["mean_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA kernel for mean reduction.
    """

    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.mean_cuda = mean_op.mean_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mean_cuda(x, self.dim)


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
