import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for KL divergence computation
kl_div_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void kl_div_kernel(const float* predictions, const float* targets, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float p = predictions[idx];
        float q = targets[idx];
        // Compute log(q) with numerical stability
        float log_q = (q > 1e-8f) ? logf(q) : -18.42069f; // log(1e-8) ≈ -18.42069
        // Compute KL divergence element-wise
        output[idx] = p * (log_q - q);
    }
}

torch::Tensor kl_div_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto size = predictions.numel();
    auto out = torch::zeros_like(predictions);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    kl_div_kernel<<<num_blocks, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );
    
    return out;
}
"""

# C++ declaration for the CUDA function
kl_div_cpp_source = (
    "torch::Tensor kl_div_cuda(torch::Tensor predictions, torch::Tensor targets);"
)

# Compile the inline CUDA code
kl_div_op = load_inline(
    name="kl_div",
    cpp_sources=kl_div_cpp_source,
    cuda_sources=kl_div_cuda_source,
    functions=["kl_div_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.kl_div_op = kl_div_op

    def forward(self, predictions, targets):
        # Ensure log is applied to predictions
        log_predictions = torch.log(predictions)
        # Use custom CUDA operator for KL divergence
        return self.kl_div_op.kl_div_cuda(log_predictions, targets).mean(dim=0).sum()
