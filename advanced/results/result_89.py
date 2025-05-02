import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel for parallel prefix sum (cumsum)
prefix_sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Simple inclusive prefix sum on 1D array using shared memory
__global__ void prefix_sum_kernel(const float* input, float* output, int size) {
    __shared__ float s_data[256];
    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (global_idx < size) {
        s_data[tid] = input[global_idx];
    } else {
        s_data[tid] = 0.0f;
    }

    // Perform exclusive prefix sum within warp
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        if (tid >= stride) {
            s_data[tid] += s_data[tid - stride];
        }
    }

    __syncthreads();

    // Write result back to global memory
    if (global_idx < size) {
        output[global_idx] = s_data[tid];
    }
}

torch::Tensor prefix_sum_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto output = torch::zeros_like(x);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    // Launch the kernel
    prefix_sum_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

prefix_sum_cpp_source = "torch::Tensor prefix_sum_cuda(torch::Tensor x);"

# Compile the inline CUDA code
prefix_sum_op = load_inline(
    name="prefix_sum",
    cpp_sources=prefix_sum_cpp_source,
    cuda_sources=prefix_sum_source,
    functions=["prefix_sum_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model that uses a custom CUDA kernel to perform cumulative sum (prefix sum).
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.prefix_sum = prefix_sum_op

    def forward(self, x):
        # Reshape tensor to 2D: [batch_size, feature_dim]
        batch_size = x.size(0)
        feature_dim = x.size(1)

        # Flatten all other dimensions if any (assuming dim=1 is the scan dimension)
        x_flat = x.view(batch_size, feature_dim)

        # Apply custom CUDA cumsum on each row
        results = []
        for i in range(batch_size):
            row = x_flat[i].contiguous()
            result = self.prefix_sum.prefix_sum_cuda(row)
            results.append(result)

        # Stack the results back together
        output_flat = torch.stack(results)

        # Restore original shape (except along scan dimension)
        output = output_flat.view_as(x)

        return output
