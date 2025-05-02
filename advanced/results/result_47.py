import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for sum reduction along a specified dimension
sum_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

// CUDA kernel to perform sum reduction along a specified dimension
template <typename scalar_t>
__global__ void sum_reduction_kernel(const scalar_t* input, scalar_t* output, 
                                     int input_size, int reduce_dim_size,
                                     int leading_dims, int trailing_dims) {
    // Each block reduces one (leading x trailing) segment across the reduce dimension
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= input_size / reduce_dim_size) return;

    scalar_t sum = 0;
    int base_idx = tid / trailing_dims * reduce_dim_size * trailing_dims + (tid % trailing_dims);
    for (int i = 0; i < reduce_dim_size; ++i) {
        sum += input[base_idx];
        base_idx += trailing_dims;
    }

    output[tid] = sum;
}

// Main dispatch function
torch::Tensor sum_reduction_cuda(torch::Tensor input, int dim) {
    // Ensure input is contiguous
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");

    // Get input dimensions
    auto sizes = input.sizes();
    int ndims = sizes.size();
    
    // Handle negative dimension
    dim = dim < 0 ? dim + ndims : dim;
    TORCH_CHECK(dim >= 0 && dim < ndims, "Dimension out of range");

    // Compute relevant dimension sizes
    int leading_dims = 1;
    for (int i = 0; i < dim; ++i) {
        leading_dims *= sizes[i];
    }
    int reduce_dim_size = sizes[dim];
    int trailing_dims = 1;
    for (int i = dim + 1; i < ndims; ++i) {
        trailing_dims *= sizes[i];
    }

    // Create output tensor
    std::vector<int64_t> output_sizes;
    for (int i = 0; i < ndims; ++i) {
        if (i == dim) {
            output_sizes.push_back(1);
        } else {
            output_sizes.push_back(sizes[i]);
        }
    }
    auto output = torch::empty(output_sizes, input.options());

    // Launch CUDA kernel
    const int input_size = input.numel();
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((input_size / reduce_dim_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "sum_reduction_cuda", ([&] {
        sum_reduction_kernel<scalar_t><<<grid, block>>>(
            input.data_ptr<scalar_t>(), 
            output.data_ptr<scalar_t>(),
            input_size, 
            reduce_dim_size,
            leading_dims,
            trailing_dims
        );
    }));

    return output;
}
"""

# C++ interface for our CUDA kernel
sum_reduction_cpp_source = """
torch::Tensor sum_reduction_cuda(torch::Tensor input, int dim);
"""

# Compile the inline CUDA code
sum_reduction_ops = load_inline(
    name="sum_reduction_ops",
    cpp_sources=sum_reduction_cpp_source,
    cuda_sources=sum_reduction_source,
    functions=["sum_reduction_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA kernel for sum reduction.
    """

    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.
        Args:
            dim (int): Dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.sum_reduction = sum_reduction_ops

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies sum reduction over the specified dimension using a custom CUDA kernel.
        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).
        Returns:
            torch.Tensor: Output tensor after sum reduction, shape (..., 1, ...).
        """
        return self.sum_reduction.sum_reduction_cuda(x, self.dim)


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs sum reduction over a specified dimension.
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
        Applies sum reduction over the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor after sum reduction, shape (..., 1, ...).
        """
        return torch.sum(x, dim=self.dim, keepdim=True)

batch_size = 16
dim1 = 256
dim2 = 256
reduce_dim = 1

def get_inputs():
    x = torch.randn(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return [reduce_dim]


# correctness
assert len(get_init_inputs()) == 0
inputs = get_inputs()
inputs = [x.cuda() for x in inputs]
model_result = Model()(*inputs)
model_new_result = ModelNew()(*inputs)
assert torch.allclose(model_result.detach().cpu(), model_new_result.detach().cpu(), rtol=1e-02, atol=1e-03)

# profiling
import triton.profiler as proton
from triton.testing import do_bench

def bench(func, warmup=0, repeat=10, proton_name="kernel"):
    with proton.scope(proton_name, metrics={}):
        ms = do_bench(func, warmup=warmup, rep=repeat)
    return ms

func_model = lambda: Model()(*inputs)
func_model_new = lambda: ModelNew()(*inputs)
model_ms = bench(func_model, warmup=0, repeat=10, proton_name="Model")
model_new_ms = bench(func_model_new, warmup=0, repeat=10, proton_name="ModelNew")
print(f"Model: {model_ms} ms")
print(f"ModelNew: {model_new_ms} ms")
