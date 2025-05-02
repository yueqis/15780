import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define inline CUDA source for reverse cumulative sum
reverse_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void reverse_cumsum_kernel(const float* input, float* output, int N, int dim_size, int stride, int step) {
    extern __shared__ float s_data[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Compute base index in the tensor for this thread
    int offset = (idx / dim_size) * dim_size * stride + (idx % dim_size);
    const float* in_base = input + offset;
    float* out_base = output + offset;

    // Load data into shared memory (reversed)
    for (int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        s_data[i] = in_base[(dim_size - 1 - i) * stride];
    }
    __syncthreads();

    // Perform inclusive prefix sum
    for (int i = 1; i < dim_size; ++i) {
        s_data[i] += s_data[i - 1];
    }

    // Write reversed cumsum result back
    for (int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        out_base[i * stride] = s_data[i];
    }
}

torch::Tensor reverse_cumsum_cuda(torch::Tensor input, int dim) {
    auto input_sizes = input.sizes();
    int64_t dim_size = input_sizes[dim];
    int64_t N = input.numel() / dim_size;

    auto output = torch::empty_like(input);

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    const int smem = dim_size * sizeof(float);

    // Launch kernel
    reverse_cumsum_kernel<<<blocks, threads, smem>>>(input.data_ptr<float>(), output.data_ptr<float>(),
                                                    N, dim_size, input.stride(dim), dim);
    return output;
}
"""

reverse_cumsum_cpp_source = """
torch::Tensor reverse_cumsum_cuda(torch::Tensor input, int dim);
"""

# Compile the inline CUDA code
reverse_cumsum_op = load_inline(
    name="reverse_cumsum",
    cpp_sources=reverse_cumsum_cpp_source,
    cuda_sources=reverse_cumsum_source,
    functions=["reverse_cumsum_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.reverse_cumsum = reverse_cumsum_op

    def forward(self, x):
        return self.reverse_cumsum.reverse_cumsum_cuda(x, self.dim)


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs a reverse cumulative sum operation along a specified dimension.

    Parameters:
        dim (int): The dimension along which to perform the reverse cumulative sum.
    """

    def __init__(self, dim):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cumsum(x.flip(self.dim), dim=self.dim).flip(self.dim)

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
