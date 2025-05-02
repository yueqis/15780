import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for exclusive cumulative sum
exclusive_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel to perform exclusive cumulative sum along dim=1 (assumes input is 2D)
__global__ void exclusive_cumsum_kernel(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row < rows) {
        float sum = 0.0f;
        for (int i = 0; i < cols; ++i) {
            if (i > 0) {
                output[row * cols + i] = sum;
            }
            sum += input[row * cols + i];
        }
    }
}

torch::Tensor exclusive_cumsum_cuda(torch::Tensor x) {
    int batch_size = x.size(0);
    int feature_dim = x.size(1);

    auto out = torch::zeros_like(x);

    // Launch one thread block per batch element
    dim3 blocks(batch_size);
    dim3 threads(feature_dim);

    exclusive_cumsum_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), batch_size, feature_dim);

    return out;
}
"""

exclusive_cumsum_cpp_source = "torch::Tensor exclusive_cumsum_cuda(torch::Tensor x);"

# Compile the inline CUDA code
exclusive_cumsum_op = load_inline(
    name="exclusive_cumsum",
    cpp_sources=exclusive_cumsum_cpp_source,
    cuda_sources=exclusive_cumsum_source,
    functions=["exclusive_cumsum_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model with a custom CUDA kernel for exclusive cumulative sum.
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        self.exclusive_cumsum = exclusive_cumsum_op

    def forward(self, x):
        return self.exclusive_cumsum.exclusive_cumsum_cuda(x)


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
