import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Inline CUDA kernel for optimized cumulative sum (prefix sum)
cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Parallel prefix sum kernel using a naive approach that works well for 1D per element in batch
template<int BLOCK_SIZE>
__global__ void cumsum_kernel(const float* input, float* output, int n_outer, int inner_size) {
    int outer_idx = blockIdx.x;
    if (outer_idx >= n_outer) return;

    int thread_idx = threadIdx.x;

    __shared__ float shared[BLOCK_SIZE];

    // Copy from global to shared memory in strided fashion
    int base_in = outer_idx * inner_size;
    int base_out = outer_idx * inner_size;
    int i = thread_idx;

    while (i < inner_size && i < BLOCK_SIZE) {
        shared[i] = input[base_in + i];
        i += blockDim.x;
    }

    __syncthreads();

    // Naive prefix sum in shared memory
    for (int j = 1; j < inner_size && j < BLOCK_SIZE; j++) {
        if (thread_idx >= j) {
            shared[thread_idx] += shared[thread_idx - 1];
        }
        __syncthreads();
    }

    // Write back results
    i = thread_idx;
    while (i < inner_size && i < BLOCK_SIZE) {
        output[base_out + i] = shared[i];
        i += blockDim.x;
    }
}

torch::Tensor cumsum_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0 || dim >= input.dim()) {
        throw std::invalid_argument("Invalid dimension for cumsum");
    }

    if (input.dim() != 2) {
        throw std::invalid_argument("Only 2D tensors supported for custom cumsum");
    }

    auto input_contig = input.contiguous();
    int n_outer = input_contig.size(dim);
    int inner_size = input_contig.size(1 - dim);

    auto output = torch::zeros_like(input_contig);

    const int block_size = 256;
    dim3 grid(n_outer);
    dim3 block(block_size);

    if (dim == 1) {
        AT_DISPATCH_FLOATING_TYPES(input_contig.type(), "cumsum_cuda", ([&] {
            cumsum_kernel<block_size><<<grid, block>>>(
                input_contig.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                n_outer,
                inner_size
            );
        }));
    } else {
        // For dim == 0 we need to transpose, compute, and transpose back
        auto transposed = input_contig.transpose(0, 1).contiguous();
        auto out_transposed = torch::zeros_like(transposed);
        AT_DISPATCH_FLOATING_TYPES(transposed.type(), "cumsum_cuda", ([&] {
            cumsum_kernel<block_size><<<inner_size, block>>>(
                transposed.data_ptr<scalar_t>(),
                out_transposed.data_ptr<scalar_t>(),
                inner_size,
                n_outer
            );
        }));
        output = out_transposed.transpose(0, 1).contiguous();
    }

    return output;
}
"""

cumsum_cpp_source = """
torch::Tensor cumsum_cuda(torch::Tensor input, int64_t dim);
"""

# Compile the inline CUDA code
custom_cumsum = load_inline(
    name="custom_cumsum",
    cpp_sources=cumsum_cpp_source,
    cuda_sources=cumsum_source,
    functions=["cumsum_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA implementation of cumulative sum (prefix sum) operation.

    Parameters:
        dim (int): The dimension along which to perform the scan operation.
    """

    def __init__(self, dim):
        """
        Initialize the Scan model with custom CUDA implementation.

        Args:
            dim (int): The dimension along which to perform the cumulative sum.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.cumsum_op = custom_cumsum

    def forward(self, x):
        """
        Forward pass for the Scan model, computing the cumulative sum along the specified dimension
        using a custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).

        Returns:
            torch.Tensor: Tensor of the same shape as `x` after applying cumulative sum along `dim`.
        """
        return self.cumsum_op.cumsum_cuda(x, self.dim)


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A simple model that performs a cumulative sum (prefix sum) operation along a specified dimension.

    Parameters:
        dim (int): The dimension along which to perform the scan operation.
    """

    def __init__(self, dim):
        """
        Initialize the Scan model.

        Args:
            dim (int): The dimension along which to perform the cumulative sum.
        """
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x):
        """
        Forward pass for the Scan model, computing the cumulative sum along the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape), where `*input_shape` 
                              can vary depending on the use case.

        Returns:
            torch.Tensor: Tensor of the same shape as `x` after applying cumulative sum along `dim`.
        """
        return torch.cumsum(x, dim=self.dim)

# Define input dimensions and parameters
batch_size = 128
input_shape = (4000,)  # Example shape (arbitrary)
dim = 1

def get_inputs():
    """
    Generates random inputs for testing the Scan model.

    Returns:
        list: A list containing a single randomly generated tensor with shape 
              (batch_size, *input_shape).
    """
    return [torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    """
    Returns the initialization parameters for the Scan model.

    Returns:
        list: A list containing the `dim` parameter for model initialization.
    """
    return [dim]


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
