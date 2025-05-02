import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Inline CUDA kernel for reverse cumulative sum (reverse cumsum)
reverse_cumsum_cuda_code = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel to perform reverse cumsum along the last dimension
__global__ void reverse_cumsum_kernel(const float* input, float* output, int dim_size, int stride) {
    int batch_idx = blockIdx.x;
    int feature_idx = blockIdx.y;

    const float* in = input + batch_idx * stride * dim_size + feature_idx;
    float* out = output + batch_idx * stride * dim_size + feature_idx;

    float acc = 0.0f;
    for (int i = 0; i < dim_size; ++i) {
        acc += in[(dim_size - 1 - i) * stride];
        out[i * stride] = acc;
    }
}

torch::Tensor reverse_cumsum_cuda(torch::Tensor input, int dim) {
    // Ensure the tensor is contiguous and on GPU
    TORCH_CHECK(input.is_contiguous() && input.is_cuda(), "Input must be a contiguous CUDA tensor");

    // Convert negative dims
    if (dim < 0) dim += input.dim();

    // Only support 2D tensors for now
    TORCH_CHECK(input.dim() == 2, "Only 2D tensors supported");
    TORCH_CHECK(dim == 1, "Only dim=1 supported for now");

    int64_t size[2] = {input.size(0), input.size(1)};
    auto output = torch::empty_like(input);

    // Launch kernel
    dim3 blocks(size[0], size[1]);  // batch_size x features
    reverse_cumsum_kernel<<<blocks, 1>>>(input.data_ptr<float>(), output.data_ptr<float>(), size[1], 1);
    
    return output;
}
"""

reverse_cumsum_cpp_binding = """
torch::Tensor reverse_cumsum_cuda(torch::Tensor input, int dim);
"""

# Compile the inline CUDA code
reverse_cumsum_op = load_inline(
    name="reverse_cumsum",
    cpp_sources=reverse_cumsum_cpp_binding,
    cuda_sources=reverse_cumsum_cuda_code,
    functions=["reverse_cumsum_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA kernel for reverse cumulative sum.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.reverse_cumsum = reverse_cumsum_op.reverse_cumsum_cuda

    def forward(self, x):
        return self.reverse_cumsum(x, self.dim)
