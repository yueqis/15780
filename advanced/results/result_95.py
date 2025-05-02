```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Cross Entropy Loss
cross_entropy_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void cross_entropy_loss_kernel(const float* predictions, const int64_t* targets, float* loss, int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        int target_class = targets[idx];
        const float* logits = predictions + idx * num_classes;
        
        // Compute max logit for numerical stability
        float max_logit = -INFINITY;
        for (int c = 0; c < num_classes; ++c) {
            max_logit = fmaxf(max_logit, logits[c]);
        }

        // Compute log(sum(exp(logits)))
        float log_sum_exp = 0.0f;
        for (int c = 0; c < num_classes; ++c) {
            log_sum_exp += exp(logits[c] - max_logit);
        }
        log_sum_exp = log(log_sum_exp);

        // Compute negative log likelihood for the target class
        loss[idx] = -(logits[target_class] - max_logit - log_sum_exp);
    }
}

torch::Tensor cross_entropy_loss_cuda(torch::Tensor predictions, torch::Tensor targets, int64_t batch_size, int64_t num_classes) {
    auto loss = torch::zeros({batch_size}, predictions.options());
    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;

    cross_entropy_loss_kernel<<<num_blocks, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        loss.data_ptr<float>(),
        batch_size,
        num_classes
    );
    return loss.mean();
}
"""

cross_entropy_loss_cpp_source = (
    "torch::Tensor cross_entropy_loss_cuda(torch::Tensor predictions, torch::Tensor targets, int64_t batch_size, int64_t num_classes);"
)

# Compile the inline CUDA code for cross entropy loss
cross_entropy_loss_op = load_inline(
    name="cross_entropy_loss",
    cpp_sources=cross_entropy_loss_cpp_source,
    cuda_sources=cross_entropy_loss_source,
    functions=["cross_entropy_loss_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_cuda_cflags=["-O2"]
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.cross_entropy_loss = cross_entropy_loss_op

    def forward(self, predictions, targets):
        return self.cross_entropy_loss.cross_entropy_loss_cuda(predictions, targets, predictions.size(0), predictions.size(1))
```