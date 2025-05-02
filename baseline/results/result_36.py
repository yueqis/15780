import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

rmsnorm_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void rmsnorm_kernel(const float* x, float* out, int num_features, int size, float eps) {
    const int N = num_features;
    const int rest_size = size / N;

    int feature_idx = blockIdx.x;
    int rest_idx = blockIdx.y;

    int x_start = feature_idx + rest_idx * N;
    
    // Compute mean squared value for each sample
    float ms = 0.0f;
    for (int i = 0; i < N; i++) {
        float val = x[i + rest_idx * N];
        ms += val * val;
    }
    ms /= N;
    ms += eps;
    float inv_rms = rsqrtf(ms);

    // Normalize
    for (int i = 0; i < N; i++) {
        out[i + rest_idx * N] = x[i + rest_idx * N] * inv_rms;
    }
}

torch::Tensor rmsnorm_cuda(torch::Tensor x, float eps) {
    auto size = x.numel();
    auto shape = x.sizes();
    auto out = torch::empty_like(x);
    
    int batch_dim = shape[0];
    int num_features = shape[1];
    dim3 blocks(num_features, batch_dim * (shape.size() > 2 ? shape[2] : 1) * (shape.size() > 3 ? shape[3] : 1));

    rmsnorm_kernel<<<blocks, 1>>>(x.data_ptr<float>(), out.data_ptr<float>(), num_features, size, eps);
    
    return out;
}
"""

rmsnorm_cpp_source = "torch::Tensor rmsnorm_cuda(torch::Tensor x, float eps);"

# Compile the inline CUDA code
rmsnorm_op = load_inline(
    name="rmsnorm",
    cpp_sources=rmsnorm_cpp_source,
    cuda_sources=rmsnorm_cuda_source,
    functions=["rmsnorm_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized RMSNorm layer using a custom CUDA kernel.
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.rmsnorm_cuda = rmsnorm_op.rmsnorm_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply custom CUDA RMS Normalization
        return self.rmsnorm_cuda(x, self.eps)
