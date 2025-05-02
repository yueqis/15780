import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Inline CUDA kernel for fused MSE computation
mse_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mse_kernel(const float* predictions, const float* targets, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float sdata[];
    
    if (idx < size) {
        float diff = predictions[idx] - targets[idx];
        sdata[threadIdx.x] = diff * diff;
    } else {
        sdata[threadIdx.x] = 0.0f;
    }

    __syncthreads();

    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Final warp reduce
    if (threadIdx.x < 32) {
        sdata[threadIdx.x] += sdata[threadIdx.x + 32];
        __syncwarp();
        sdata[threadIdx.x] += sdata[threadIdx.x + 16];
        __syncwarp();
        sdata[threadIdx.x] += sdata[threadIdx.x + 8];
        __syncwarp();
        sdata[threadIdx.x] += sdata[threadIdx.x + 4];
        __syncwarp();
        sdata[threadIdx.x] += sdata[threadIdx.x + 2];
        __syncwarp();
        sdata[threadIdx.x] += sdata[threadIdx.x + 1];
        __syncwarp();
    }

    if (threadIdx.x == 0) {
        atomicAdd(output, sdata[0] / size);
    }
}

torch::Tensor mse_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto size = predictions.numel();
    auto output = torch::zeros({}, predictions.options());

    const int block_size = 512;
    const int num_blocks = 1;  // Single block since we're doing a global reduction

    size_t smem_size = block_size * sizeof(float);

    mse_kernel<<<num_blocks, block_size, smem_size>>>(predictions.data_ptr<float>(),
                                                     targets.data_ptr<float>(),
                                                     output.data_ptr<float>(),
                                                     size);

    return output;
}
"""

mse_cpp_source = (
    "torch::Tensor mse_cuda(torch::Tensor predictions, torch::Tensor targets);"
)

# Compile the inline CUDA code
mse_op = load_inline(
    name="mse_op",
    cpp_sources=mse_cpp_source,
    cuda_sources=mse_cuda_source,
    functions=["mse_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.mse_op = mse_op

    def forward(self, predictions, targets):
        return self.mse_op.mse_cuda(predictions, targets)
