import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Inline CUDA kernel for optimized Hinge Loss computation
hinge_loss_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hinge_loss_kernel(const float* predictions, const float* targets, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float product = predictions[idx] * targets[idx];
        float loss = fmaxf(1.0f - product, 0.0f);
        out[idx] = loss;
    }
}

torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto size = predictions.numel();
    auto out = torch::zeros_like(predictions);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    hinge_loss_kernel<<<num_blocks, block_size>>>(
        predictions.data_ptr<float>(), 
        targets.data_ptr<float>(), 
        out.data_ptr<float>(), 
        size
    );

    return out;
}
"""

hinge_loss_cpp_source = (
    "torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets);"
)

# Compile the inline CUDA code
hinge_loss_op = load_inline(
    name="hinge_loss",
    cpp_sources=hinge_loss_cpp_source,
    cuda_sources=hinge_loss_cuda_source,
    functions=["hinge_loss_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA kernel for Hinge Loss computation.
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        self.hinge_loss_op = hinge_loss_op

    def forward(self, predictions, targets):
        loss_tensor = self.hinge_loss_op.hinge_loss_cuda(predictions, targets)
        return torch.mean(loss_tensor)
