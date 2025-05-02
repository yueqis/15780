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
