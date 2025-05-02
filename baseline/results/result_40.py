import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Inline CUDA kernel for LayerNorm
layernorm_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void layernorm_kernel(
    const scalar_t* input,
    scalar_t* output,
    const scalar_t* weight,
    const scalar_t* bias,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W) {

    int64_t sample_idx = blockIdx.x;
    int64_t channel_idx = blockIdx.y;

    int64_t pos = sample_idx * C * H * W + channel_idx * H * W;

    // Compute mean
    double mean = 0.0;
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            mean += input[pos + h * W + w];
        }
    }
    mean /= (H * W);

    // Compute variance
    double var = 0.0;
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            double x = input[pos + h * W + w] - mean;
            var += x * x;
        }
    }
    var /= (H * W);

    double inv_std = 1.0 / sqrt(var + 1e-5);

    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            double x_centered = input[pos + h * W + w] - mean;
            output[pos + h * W + w] = static_cast<scalar_t>(
                (x_centered * inv_std) * weight[channel_idx] + bias[channel_idx]);
        }
    }
}

torch::Tensor layernorm_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto size = input.sizes();
    int64_t N = size[0];
    int64_t C = size[1];
    int64_t H = size[2];
    int64_t W = size[3];

    auto output = torch::empty_like(input);

    dim3 grid(N, C);
    dim3 block(1);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "layernorm_cuda", ([&] {
        layernorm_kernel<scalar_t><<<grid, block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            N, C, H, W);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\\n", cudaGetErrorString(err));
        TORCH_CHECK(false, "CUDA error in layernorm kernel");
    }

    return output;
}
"""

layernorm_cpp_source = """
torch::Tensor layernorm_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the inline CUDA code
layernorm_op = load_inline(
    name="layernorm",
    cpp_sources=layernorm_cpp_source,
    cuda_sources=layernorm_cuda_source,
    functions=["layernorm_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA implementation of LayerNorm.
    """

    def __init__(self, normalized_shape: tuple):
        super(ModelNew, self).__init__()
        self.normalized_shape = normalized_shape
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.layernorm_cuda = layernorm_op.layernorm_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape to ensure 4D tensor (batch_size, channels, height, width)
        original_shape = x.shape
        batch_size = x.size(0)
        feature_dim = x.size(1)
        spatial_dims = x.size()[2:]

        x = x.view(batch_size, feature_dim, *spatial_dims)

        # Apply custom CUDA LayerNorm
        output = self.layernorm_cuda(x, self.weight, self.bias)

        # Restore original shape
        return output.view(original_shape)


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
