import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA kernel for Instance Normalization
instance_norm_cuda_code = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

// Compute mean and inverse standard deviation per instance (feature map)
__global__ void instance_norm_forward_kernel(
    const float* input,
    float* output,
    const float* weight,  // gamma
    const float* bias,    // beta
    int batch_size,
    int num_features,
    int height,
    int width) {

    int n = blockIdx.z; // batch index
    int c = blockIdx.y; // feature/channel index

    int spatial_size = height * width;
    int total_size = batch_size * num_features * spatial_size;

    const float eps = 1e-5;

    // Compute mean
    float mean = 0.0f;
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            int idx = n * num_features * spatial_size + c * spatial_size + h * width + w;
            mean += input[idx];
        }
    }
    mean /= spatial_size;

    // Compute variance
    float var = 0.0f;
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            int idx = n * num_features * spatial_size + c * spatial_size + h * width + w;
            float diff = input[idx] - mean;
            var += diff * diff;
        }
    }
    var /= spatial_size;

    // Compute inverse std dev with epsilon
    float inv_std = 1.0f / sqrtf(var + eps);

    // Normalize
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            int idx = n * num_features * spatial_size + c * spatial_size + h * width + w;
            float normalized = (input[idx] - mean) * inv_std;

            // Apply scale and shift if weights are provided
            if (weight && bias) {
                output[idx] = normalized * weight[c] + bias[c];
            } else {
                output[idx] = normalized;
            }
        }
    }
}

// Wrapper function to call the CUDA kernel
torch::Tensor instance_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) {

    int batch_size = input.size(0);
    int num_features = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    auto output = torch::empty_like(input);

    dim3 blocks(1, num_features, batch_size);  // One block per feature map

    instance_norm_forward_kernel<<<blocks, 1>>>( // Single thread per block handles entire feature map
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        weight ? weight.data_ptr<float>() : nullptr,
        bias ? bias.data_ptr<float>() : nullptr,
        batch_size,
        num_features,
        height,
        width);

    return output;
}
"""

# C++ declaration for CUDA kernel
instance_norm_cpp_code = """
torch::Tensor instance_norm_forward_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the inline CUDA code
instance_norm_op = load_inline(
    name="instance_norm",
    cpp_sources=instance_norm_cpp_code,
    cuda_sources=instance_norm_cuda_code,
    functions=["instance_norm_forward_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA kernel for Instance Normalization.
    """

    def __init__(self, num_features: int):
        """
        Initializes the InstanceNorm layer.
        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))
        self.num_features = num_features

        # Initialize parameters
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

        # Load custom operator
        self.instance_norm_op = instance_norm_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Instance Normalization to the input tensor using a custom CUDA kernel.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).
        Returns:
            torch.Tensor: Output tensor with Instance Normalization applied, same shape as input.
        """
        return self.instance_norm_op.instance_norm_forward_cuda(
            x, self.weight, self.bias
        )
