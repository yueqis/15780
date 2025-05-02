import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel for performing exclusive cumulative sum
exclusive_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Custom CUDA kernel to compute exclusive cumulative sum along a given dimension
__global__ void exclusive_cumsum_kernel(const float* input, float* output, int size, int dim_size, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int offset = idx / dim_size;
        int inner_idx = idx % dim_size;

        // Exclusive cumulative sum logic: copy previous elements and accumulate
        for (int i = 0; i < inner_idx; ++i) {
            output[offset * dim_size + inner_idx] += input[offset * dim_size + i];
        }
        // Set first element in each row to zero
        if (inner_idx == 0) {
            output[offset * dim_size + inner_idx] = 0;
        }
    }
}

torch::Tensor exclusive_cumsum_cuda(torch::Tensor x, int dim) {
    // Ensure the tensor is contiguous and on GPU
    TORCH_CHECK(x.is_contiguous() && x.is_cuda(), "Input must be contiguous and on GPU");

    int64_t num_dims = x.dim();
    TORCH_CHECK(dim >= 0 && dim < num_dims, "Invalid dimension for cumsum");

    // Get sizes and strides
    auto sizes = x.sizes();
    int64_t total_size = 1;
    int64_t dim_size = sizes[dim];
    for (int64_t i = 0; i < num_dims; ++i) {
        total_size *= sizes[i];
    }

    // Compute stride for collapsing other dimensions
    int64_t stride = 1;
    for (int64_t i = dim + 1; i < num_dims; ++i) {
        stride *= sizes[i];
    }

    // Prepare output tensor
    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    torch::Tensor out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;

    exclusive_cumsum_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), 
        out.data_ptr<float>(), 
        total_size, 
        dim_size, 
        stride
    );

    return out;
}
"""

exclusive_cumsum_cpp_source = """
torch::Tensor exclusive_cumsum_cuda(torch::Tensor x, int dim);
"""

# Compile the inline CUDA code
exclusive_cumsum_op = load_inline(
    name="exclusive_cumsum",
    cpp_sources=exclusive_cumsum_cpp_source,
    cuda_sources=exclusive_cumsum_source,
    functions=["exclusive_cumsum_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA kernel to perform an exclusive cumulative sum.
    Parameters:
        dim (int): The dimension along which to perform the exclusive cumulative sum.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.exclusive_cumsum = exclusive_cumsum_op

    def forward(self, x):
        return self.exclusive_cumsum.exclusive_cumsum_cuda(x, self.dim)


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs an exclusive cumulative sum (does not include the current element).

    Parameters:
        dim (int): The dimension along which to perform the exclusive cumulative sum.
    """

    def __init__(self, dim):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x):
        exclusive_cumsum = torch.cat((torch.zeros_like(x.select(self.dim, 0).unsqueeze(self.dim)), x), dim=self.dim)[:-1]
        return torch.cumsum(exclusive_cumsum, dim=self.dim)

batch_size = 128
input_shape = (4000,)
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape)]

def get_init_inputs():
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
