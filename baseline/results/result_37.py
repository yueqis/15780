import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for Frobenius norm normalization
frobenius_norm_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Kernel to compute squared Frobenius norm (sum of squares)
__global__ void compute_squared_norm_kernel(const float* x, float* norm_sq, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        atomicAdd(norm_sq, x[idx] * x[idx]);
    }
}

// Kernel to normalize using the computed norm
__global__ void normalize_kernel(const float* x, float* out, float norm, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = x[idx] / norm;
    }
}

torch::Tensor frobenius_norm_cuda(torch::Tensor x) {
    auto size = x.numel();
    
    // Tensor to store result
    auto out = torch::zeros_like(x);
    
    // Device scalar to hold the squared norm
    auto norm_sq = torch::zeros({1}, x.options().dtype(torch::kFloat));
    
    // Allocate raw pointers for CUDA access
    const float* x_data = x.data_ptr<float>();
    float* norm_sq_data = norm_sq.data_ptr<float>();
    
    // Set up blocks and threads for sum reduction
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    // Compute squared norm
    compute_squared_norm_kernel<<<num_blocks, block_size>>>(x_data, norm_sq_data, size);
    
    // Synchronize to ensure computation is done
    cudaDeviceSynchronize();
    
    // Take square root to get final Frobenius norm
    float norm = sqrt(norm_sq.item<float>());
    
    // Avoid division by zero
    if (norm == 0) {
        return out;  // Return zero tensor
    }

    // Get data pointer for output
    float* out_data = out.data_ptr<float>();

    // Normalize the tensor
    normalize_kernel<<<num_blocks, block_size>>>(x_data, out_data, norm, size);

    return out;
}
"""

frobenius_norm_cpp_source = "torch::Tensor frobenius_norm_cuda(torch::Tensor x);"

# Compile the inline CUDA code
frobenius_norm_op = load_inline(
    name="frobenius_norm",
    cpp_sources=frobenius_norm_cpp_source,
    cuda_sources=frobenius_norm_cuda_source,
    functions=["frobenius_norm_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized version of Model that uses a custom CUDA kernel for Frobenius norm normalization.
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        self.frobenius_norm = frobenius_norm_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.frobenius_norm.frobenius_norm_cuda(x)


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Frobenius norm normalization.
    """
    def __init__(self):
        """
        Initializes the Frobenius norm normalization layer.
        """
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Frobenius norm normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with Frobenius norm normalization applied, same shape as input.
        """
        norm = torch.norm(x, p='fro')
        return x / norm

batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
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
