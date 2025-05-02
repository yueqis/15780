import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for Layer Normalization
layer_norm_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void layer_norm_kernel(const float* x, float* out, const float* weight, const float* bias,
                                  int N, int C, int H, int W, float eps) {
    int n = blockIdx.x;
    int c = threadIdx.x;

    // Compute mean
    float mean = 0.0f;
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            int idx = n * C * H * W + c * H * W + h * W + w;
            mean += x[idx];
        }
    }
    mean /= H * W;

    // Compute variance
    float var = 0.0f;
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            int idx = n * C * H * W + c * H * W + h * W + w;
            float diff = x[idx] - mean;
            var += diff * diff;
        }
    }
    var /= H * W;

    // Normalize and apply scale and shift
    float inv_std = 1.0f / sqrt(var + eps);
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            int idx = n * C * H * W + c * H * W + h * W + w;
            float centered = x[idx] - mean;
            float normalized = centered * inv_std;
            out[idx] = weight[c] * normalized + bias[c];
        }
    }
}

torch::Tensor layer_norm_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
                              torch::IntArrayRef normalized_shape, double eps) {
    int N = x.size(0);
    int C = normalized_shape[0];
    int H = normalized_shape[1];
    int W = normalized_shape[2];

    auto options = x.options();
    torch::Tensor out = torch::empty_like(x, options);

    dim3 grid(N);
    dim3 block(C);

    layer_norm_kernel<<<grid, block>>>(
        x.data_ptr<float>(), out.data_ptr<float>(),
        weight.data_ptr<float>(), bias.data_ptr<float>(),
        N, C, H, W, static_cast<float>(eps)
    );

    return out;
}
"""

layer_norm_cpp_source = """
torch::Tensor layer_norm_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
                              torch::IntArrayRef normalized_shape, double eps);
"""

# Compile the inline CUDA code
layer_norm_op = load_inline(
    name="layer_norm",
    cpp_sources=layer_norm_cpp_source,
    cuda_sources=layer_norm_cuda_source,
    functions=["layer_norm_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super(ModelNew, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)
        self.layer_norm_cuda = layer_norm_op.layer_norm_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized_shape = tuple(
            x.shape[1:]
        )  # Assume input shape is (batch, channels, height, width)
        return self.layer_norm_cuda(
            x, self.ln.weight, self.ln.bias, normalized_shape, self.ln.eps
        )


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Layer Normalization.
    """
    def __init__(self, normalized_shape: tuple):
        """
        Initializes the LayerNorm layer.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(Model, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Layer Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        return self.ln(x)

batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [(features, dim1, dim2)]
