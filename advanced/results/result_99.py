import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA kernel for Triplet Margin Loss
triplet_margin_loss_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void triplet_margin_loss_kernel(
    const float* anchor, 
    const float* positive, 
    const float* negative, 
    float* output, 
    int64_t batch_size, 
    int64_t embedding_dim,
    float margin) {
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    float d_pos = 0.0f;
    float d_neg = 0.0f;

    for (int64_t j = 0; j < embedding_dim; ++j) {
        float a_j = anchor[idx * embedding_dim + j];
        float p_j = positive[idx * embedding_dim + j];
        float n_j = negative[idx * embedding_dim + j];

        float diff_pos = a_j - p_j;
        float diff_neg = a_j - n_j;

        d_pos += diff_pos * diff_pos;
        d_neg += diff_neg * diff_neg;
    }

    float loss = fmaxf(margin + d_pos - d_neg, 0.0f);
    atomicAdd(output, loss);
}

torch::Tensor triplet_margin_loss_cuda(
    torch::Tensor anchor, 
    torch::Tensor positive, 
    torch::Tensor negative, 
    float margin) {

    auto batch_size = anchor.size(0);
    auto embedding_dim = anchor.size(1);

    auto output = torch::zeros({}, anchor.options());

    int64_t threads_per_block = 256;
    int64_t blocks = (batch_size + threads_per_block - 1) / threads_per_block;

    triplet_margin_loss_kernel<<<blocks, threads_per_block>>>(
        anchor.data_ptr<float>(), 
        positive.data_ptr<float>(), 
        negative.data_ptr<float>(),
        output.data_ptr<float>(), 
        batch_size, 
        embedding_dim, 
        margin);

    output /= batch_size;

    return output;
}
"""

triplet_margin_loss_cpp_source = """
torch::Tensor triplet_margin_loss_cuda(torch::Tensor anchor, torch::Tensor positive, torch::Tensor negative, float margin);
"""

# Compile the inline CUDA code
triplet_margin_loss_op = load_inline(
    name="triplet_margin_loss",
    cpp_sources=triplet_margin_loss_cpp_source,
    cuda_sources=triplet_margin_loss_cuda_source,
    functions=["triplet_margin_loss_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin
        self.loss_fn = triplet_margin_loss_op.triplet_margin_loss_cuda

    def forward(self, anchor, positive, negative):
        return self.loss_fn(anchor, positive, negative, self.margin)
