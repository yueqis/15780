import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for L1 normalization
l1_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void l1_norm_kernel(const float* x, float* out, int rows, int cols) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    extern __shared__ float shared_abs_sum[];

    // Compute L1 norm per row
    float abs_sum = 0.0f;
    for (int i = col; i < cols; i += blockDim.x) {
        int idx = row * cols + i;
        abs_sum += fabsf(x[idx]);
    }

    // Reduce within the thread block
    shared_abs_sum[col] = abs_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (col < s) {
            shared_abs_sum[col] += shared_abs_sum[col + s];
        }
        __syncthreads();
    }

    // Normalize input values
    if (col == 0) {
        abs_sum = shared_abs_sum[0];
        // Avoid division by zero
        if (abs_sum == 0.0f) abs_sum = 1.0f;
    }
    __syncthreads();

    for (int i = col; i < cols; i += blockDim.x) {
        int idx = row * cols + i;
        out[idx] = x[idx] / shared_abs_sum[0];
    }
}

torch::Tensor l1_norm_cuda(torch::Tensor x) {
    int batch_size = x.size(0);
    int dim = x.size(1);

    auto out = torch::zeros_like(x);

    int threads = 256;
    int shared_mem = threads * sizeof(float);

    l1_norm_kernel<<<batch_size, threads, shared_mem>>>(x.data_ptr<float>(), out.data_ptr<float>(), batch_size, dim);

    return out;
}
"""

l1_norm_cpp_source = "torch::Tensor l1_norm_cuda(torch::Tensor x);"

# Compile the inline CUDA code
l1_norm_op = load_inline(
    name="l1_norm",
    cpp_sources=l1_norm_cpp_source,
    cuda_sources=l1_norm_source,
    functions=["l1_norm_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA kernel for L1 normalization.
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        self.l1_norm = l1_norm_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous and move to GPU
        x = x.contiguous().cuda()
        # Use custom CUDA implementation for L1 normalization
        return self.l1_norm.l1_norm_cuda(x)


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs L1 normalization.
    """
    def __init__(self):
        """
        Initializes the L1 normalization layer.
        """
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L1 normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor with L1 normalization applied, same shape as input.
        """
        return x / torch.sum(torch.abs(x), dim=1, keepdim=True)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []


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
