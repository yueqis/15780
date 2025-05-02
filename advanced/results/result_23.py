import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Softmax
softmax_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* input, float* output, int batch_size, int dim) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    int index = row * dim + tid;

    // Shared memory to hold intermediate values
    extern __shared__ float s_data[];
    
    // Load input into shared memory
    s_data[tid] = input[index];
    __syncthreads();

    // Find max in the row
    if (dim > 1) {
        for (int stride = 1; stride < dim; stride *= 2) {
            if (tid % (2 * stride) == 0) {
                int j = tid + stride;
                if (j < dim) {
                    s_data[tid] = (s_data[tid] > s_data[j]) ? s_data[tid] : s_data[j];
                }
            }
            __syncthreads();
        }
    }

    float max_val = s_data[0];

    // Compute exponentials and sum
    float exp_val = expf(input[index] - max_val);
    output[index] = exp_val;

    // Reduce sum of exponentials
    if (dim > 1) {
        for (int stride = 1; stride < dim; stride *= 2) {
            if (tid % (2 * stride) == 0) {
                int j = tid + stride;
                if (j < dim) {
                    output[row * dim + tid] += output[row * dim + j];
                }
            }
            __syncthreads();
        }
    }

    float sum_exp = output[row * dim];

    // Normalize
    output[index] = exp_val / sum_exp;
}

torch::Tensor softmax_cuda(torch::Tensor input, int batch_size, int dim) {
    auto output = torch::zeros_like(input);
    const int block_size = 512;
    const int num_blocks = batch_size;

    softmax_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), batch_size, dim);

    return output;
}
"""

softmax_cpp_source = (
    "torch::Tensor softmax_cuda(torch::Tensor input, int batch_size, int dim);"
)

# Compile the inline CUDA code for softmax
softmax_ops = load_inline(
    name="softmax_cuda",
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_cuda_source,
    functions=["softmax_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, batch_size=16, dim=16384):
        super(ModelNew, self).__init__()
        self.batch_size = batch_size
        self.dim = dim
        self.softmax_cuda = softmax_ops.softmax_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax_cuda(x, self.batch_size, self.dim)


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a Softmax activation.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softmax activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        return torch.softmax(x, dim=1)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed


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
