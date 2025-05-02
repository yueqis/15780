import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for max reduction along a specific dimension
max_reduce_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel to compute max along dim=1 (middle dimension)
__global__ void max_reduce_kernel(const float* input, float* output, int batch_size, int dim1, int dim2) {
    int b = blockIdx.x;
    int feature_idx = threadIdx.x;
    int stride = blockDim.x;

    // Shared memory to store intermediate max values per thread block
    __shared__ float shared_max[256];

    float max_val = -1e20f;  // Initialize to negative infinity

    // Iterate over dim1 (middle dimension) in steps of stride
    for (int i = feature_idx; i < dim1; i += stride) {
        float val = input[b * dim1 * dim2 + i * dim2 + threadIdx.y];
        max_val = fmaxf(max_val, val);
    }

    // Reduce within thread block using shared memory
    shared_max[threadIdx.x] = max_val;
    __syncthreads();

    // Final reduction across threads
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + s]);
        }
        __syncthreads();
    }

    // Write final result for this batch and feature
    if (threadIdx.x == 0) {
        output[b * dim2 + threadIdx.y] = shared_max[0];
    }
}

torch::Tensor max_reduce_cuda(torch::Tensor input, int dim) {
    int batch_size = input.size(0);
    int dim1 = input.size(1);
    int dim2 = input.size(2);

    auto output = torch::zeros({batch_size, dim2}, input.options());

    // We reduce along dim=1 (middle dimension), so we fix that logic here
    if (dim != 1) {
        throw std::runtime_error("Only dim=1 supported for custom max reduction");
    }

    dim3 threads(256, 1);  // Use flat threading model for simplicity
    dim3 blocks(batch_size * dim2);

    max_reduce_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, dim1, dim2);

    return output;
}
"""

max_reduce_cpp_source = """
torch::Tensor max_reduce_cuda(torch::Tensor input, int dim);
"""

# Compile the inline CUDA code
max_reduce_op = load_inline(
    name="max_reduce",
    cpp_sources=max_reduce_cpp_source,
    cuda_sources=max_reduce_source,
    functions=["max_reduce_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA kernel to perform Max reduction over dimension 1.
    """

    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.max_reduce = max_reduce_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Only support reduction over dim=1 in this optimized version
        if self.dim != 1:
            raise ValueError("Custom kernel only supports reduction over dimension 1")
        return self.max_reduce.max_reduce_cuda(x, self.dim)
