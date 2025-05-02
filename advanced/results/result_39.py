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
