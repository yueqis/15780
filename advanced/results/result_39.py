import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for L2 normalization
l2norm_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void l2norm_kernel(const float* x, float* out, int dim_size, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        // Compute the starting index of the vector in the batch
        int vec_idx = (idx / dim_size) * dim_size;
        // Compute the norm of the vector
        float norm = 0.0f;
        for (int i = 0; i < dim_size; ++i) {
            float val = x[vec_idx + i];
            norm += val * val;
        }
        norm = sqrtf(norm);
        // Normalize each element of the vector
        for (int i = 0; i < dim_size; ++i) {
            out[vec_idx + i] = x[vec_idx + i] / (norm + 1e-12f); // Add small epsilon to avoid division by zero
        }
    }
}

torch::Tensor l2norm_cuda(torch::Tensor x) {
    auto size = x.sizes();
    auto total_size = x.numel();
    auto dim_size = size[1]; // Dimension along which to normalize
    auto out = torch::zeros_like(x);
    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;
    l2norm_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), dim_size, total_size);
    return out;
}
"""

# C++ declaration for the CUDA kernel
l2norm_cpp_source = "torch::Tensor l2norm_cuda(torch::Tensor x);"

# Compile the inline CUDA code
l2norm_op = load_inline(
    name="l2norm",
    cpp_sources=l2norm_cpp_source,
    cuda_sources=l2norm_cuda_source,
    functions=["l2norm_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.l2norm = l2norm_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2norm.l2norm_cuda(x)


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs L2 normalization.
    """
    def __init__(self):
        """
        Initializes the L2Norm layer.

        Args:
            dim (int): Dimension along which to normalize.
        """
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L2 normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (*, dim, *).

        Returns:
            torch.Tensor: Output tensor with L2 normalization applied, same shape as input.
        """
        return x / torch.norm(x, p=2, dim=1, keepdim=True)

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
