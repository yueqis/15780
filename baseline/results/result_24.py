import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA implementation of LogSoftmax
logsoftmax_cuda_code = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void log_softmax_kernel(float* x, float* output, int rows, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int idx = row * cols + tid;

    extern __shared__ float s_data[];
    
    // Load data into shared memory
    if (tid < cols) {
        s_data[tid] = x[idx];
    }
    __syncthreads();
    
    // Find max in the row
    for (int stride = cols / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_data[tid] = fmaxf(s_data[tid], s_data[tid + stride]);
        }
        __syncthreads();
    }
    float row_max = s_data[0];

    // Compute exp and sum
    float sum_exp = 0.0f;
    if (tid < cols) {
        float exp_val = expf(x[idx] - row_max);
        s_data[tid] = exp_val;
        sum_exp += exp_val;
    }
    __syncthreads();

    // Reduce sum_exp
    for (int stride = cols / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }
    float total_sum = s_data[0];

    // Compute log softmax
    if (tid < cols) {
        output[idx] = x[idx] - row_max - logf(total_sum + 1e-20f);
    }
}

torch::Tensor log_softmax_cuda(torch::Tensor x) {
    int rows = x.size(0);
    int cols = x.size(1);
    
    auto output = torch::zeros_like(x);
    
    dim3 grid(rows);
    dim3 block(BLOCK_SIZE);
    
    log_softmax_kernel<<<grid, block, cols * sizeof(float), at::cuda::getCurrentCUDAStream()>>>( 
        x.data_ptr<float>(), 
        output.data_ptr<float>(), 
        rows, 
        cols);
    
    return output;
}
"""

logsoftmax_cpp_code = "torch::Tensor log_softmax_cuda(torch::Tensor x);"

# Compile the inline CUDA code
log_softmax_op = load_inline(
    name="log_softmax",
    cpp_sources=logsoftmax_cpp_code,
    cuda_sources=logsoftmax_cuda_code,
    functions=["log_softmax_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA kernel for LogSoftmax activation.
    """

    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.log_softmax = log_softmax_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized LogSoftmax activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with LogSoftmax applied, same shape as input.
        """
        return self.log_softmax.log_softmax_cuda(x)
