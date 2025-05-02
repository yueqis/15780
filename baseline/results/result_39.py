```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for L2 normalization
l2norm_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void l2norm_kernel(const float* x, float* out, int rows, int cols) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    
    if (row < rows && col < cols) {
        // First pass: compute norm
        float sum_sq = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float val = x[row * cols + c];
            sum_sq += val * val;
        }
        
        float inv_norm = rsqrtf(sum_sq + 1e-12f);  // Add small epsilon to avoid division by zero
        
        // Second pass: normalize
        out[row * cols + col] = x[row * cols + col] * inv_norm;
    }
}

torch::Tensor l2norm_cuda(torch::Tensor x) {
    int rows = x.size(0);
    int cols = x.numel() / x.size(0);

    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int max_threads_per_block = 256;

    dim3 grid(rows);
    dim3 block(min(cols, max_threads_per_block));

    l2norm_kernel<<<grid, block>>>(x.data_ptr<float>(), out.data_ptr<float>(), rows, cols);

    return out;
}
"""

l2norm_cpp_source = "torch::Tensor l2norm_cuda(torch::Tensor x);"

# Compile the inline CUDA code for L2 normalization
l2norm_extension = load_inline(
    name="l2norm",
    cpp_sources=l2norm_cpp_source,
    cuda_sources=l2norm_cuda_source,
    functions=["l2norm_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.l2norm = l2norm_extension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2norm.l2norm_cuda(x)
```