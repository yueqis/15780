import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for cosine similarity loss: cosine_similarity + mean(1 - similarity)
cosine_loss_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <math.h>

// Computes cosine similarity and 1 - similarity in a single pass, then sums for mean
__global__ void cosine_similarity_loss_kernel(
    const float* predictions,
    const float* targets,
    const int D,
    const int N,
    float* loss_sum) {

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    float dot = 0.0f;
    float norm_pred = 0.0f;
    float norm_target = 0.0f;

    for (int d = 0; d < D; ++d) {
        int idx = n * D + d;
        float p = predictions[idx];
        float t = targets[idx];
        dot += p * t;
        norm_pred += p * p;
        norm_target += t * t;
    }

    float eps = 1e-8f;
    float denom = sqrt(norm_pred) * sqrt(norm_target) + eps;
    float sim = dot / denom;
    float loss = 1.0f - sim;

    // Use atomic add to accumulate sum on GPU memory
    atomicAdd(loss_sum, loss);
}

torch::Tensor cosine_similarity_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    int N = predictions.size(0);
    int D = predictions.size(1);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto loss_sum = torch::zeros({1}, options);

    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    cosine_similarity_loss_kernel<<<gridSize, blockSize>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        D,
        N,
        loss_sum.data_ptr<float>()
    );

    // Compute mean
    return loss_sum / static_cast<float>(N);
}
"""

cosine_loss_cpp_source = "torch::Tensor cosine_similarity_loss_cuda(torch::Tensor predictions, torch::Tensor targets);"

# Compile the inline CUDA code
cosine_loss_op = load_inline(
    name="cosine_similarity_loss",
    cpp_sources=cosine_loss_cpp_source,
    cuda_sources=cosine_loss_cuda_source,
    functions=["cosine_similarity_loss_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.cosine_loss = cosine_loss_op

    def forward(self, predictions, targets):
        return self.cosine_loss.cosine_similarity_loss_cuda(predictions, targets)
