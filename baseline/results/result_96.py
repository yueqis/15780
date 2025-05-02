import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Inline CUDA kernel for Smooth L1 (Huber) Loss
smooth_l1_loss_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void smooth_l1_loss_kernel(const float* predictions, const float* targets,
                                     float* output, int size, float beta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float residual = fabsf(predictions[idx] - targets[idx]);
        if (residual < beta) {
            output[idx] = 0.5f * residual * residual / beta;
        } else {
            output[idx] = residual - 0.5f * beta;
        }
    }
}

torch::Tensor smooth_l1_loss_cuda(torch::Tensor predictions, torch::Tensor targets, float beta) {
    auto size = predictions.numel();
    auto out = torch::zeros_like(predictions);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    smooth_l1_loss_kernel<<<num_blocks, block_size>>>(
        predictions.data_ptr<float>(), targets.data_ptr<float>(),
        out.data_ptr<float>(), size, beta);

    return out;
}
"""

smooth_l1_loss_cpp_source = """
torch::Tensor smooth_l1_loss_cuda(torch::Tensor predictions, torch::Tensor targets, float beta);
"""

# Compile the inline CUDA code
smooth_l1_loss_op = load_inline(
    name="smooth_l1_loss",
    cpp_sources=smooth_l1_loss_cpp_source,
    cuda_sources=smooth_l1_loss_cuda_source,
    functions=["smooth_l1_loss_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA kernel for Smooth L1 (Huber) Loss.
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        self.smooth_l1_loss = smooth_l1_loss_op.smooth_l1_loss_cuda

    def forward(self, predictions, targets):
        return self.smooth_l1_loss(predictions, targets, beta=1.0)
