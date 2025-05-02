import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for LogSoftmax
log_softmax_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void log_softmax_kernel(const float* x, float* out, int batch_size, int dim) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float s_data[];

    // Load the max value for the row
    float max_val = -FLT_MAX;
    for (int i = tid; i < dim; i += blockDim.x) {
        float val = x[row * dim + i];
        max_val = fmaxf(max_val, val);
    }
    __syncthreads();

    // Reduce to find max per row
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] = fmaxf(s_data[tid], s_data[tid + stride]);
        }
        __syncthreads();
    }
    max_val = s_data[0];
    __syncthreads();

    // Compute exponentials and sum
    float exp_sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float exp_val = expf(x[row * dim + i] - max_val);
        out[row * dim + i] = exp_val;
        exp_sum += exp_val;
    }
    __syncthreads();

    // Reduce to compute total sum
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }
    exp_sum = s_data[0];
    __syncthreads();

    // Compute log(softmax)
    float log_sum = logf(exp_sum);
    for (int i = tid; i < dim; i += blockDim.x) {
        out[row * dim + i] = x[row * dim + i] - max_val - log_sum;
    }
}

torch::Tensor log_softmax_cuda(torch::Tensor x, int dim) {
    int64_t batch_size = x.size(0);
    int64_t feature_dim = x.size(1);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor out = torch::empty({batch_size, feature_dim}, options);

    dim3 blocks(batch_size);
    dim3 threads(256); // Can be tuned based on GPU architecture
    size_t shared_mem = threads.x * sizeof(float);

    log_softmax_kernel<<<blocks, threads, shared_mem>>>(x.data_ptr<float>(), out.data_ptr<float>(), batch_size, feature_dim);
    return out;
}
"""

log_softmax_cpp_source = """
torch::Tensor log_softmax_cuda(torch::Tensor x, int dim);
"""

# Compile the inline CUDA code for LogSoftmax
log_softmax_op = load_inline(
    name="log_softmax",
    cpp_sources=log_softmax_cpp_source,
    cuda_sources=log_softmax_cuda_source,
    functions=["log_softmax_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.log_softmax = log_softmax_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.log_softmax.log_softmax_cuda(x, self.dim)
