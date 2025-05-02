import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for BatchNorm2d
batchnorm2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void batchnorm2d_kernel(
    const float* input, 
    const float* weight, 
    const float* bias, 
    const float* running_mean, 
    const float* running_var,
    float* output, 
    int num_features, 
    int height, 
    int width, 
    float eps) {
    
    int n = blockIdx.z;
    int c = blockIdx.y;
    int h = threadIdx.x;
    int w = threadIdx.y;

    int spatial_size = height * width;
    int idx = n * num_features * spatial_size + c * spatial_size + h * width + w;

    if (h < height && w < width) {
        float mean = running_mean[c];
        float var = running_var[c];
        float inv_std = 1.0f / sqrtf(var + eps);
        float x_hat = (input[idx] - mean) * inv_std;
        output[idx] = weight[c] * x_hat + bias[c];
    }
}

torch::Tensor batchnorm2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps) {

    int batch_size = input.size(0);
    int num_features = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    auto output = torch::zeros_like(input);

    dim3 blocks(1, num_features, batch_size);
    dim3 threads(height, width);

    batchnorm2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        num_features,
        height,
        width,
        eps);

    return output;
}
"""

batchnorm2d_cpp_source = """
torch::Tensor batchnorm2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps);
"""

# Compile the inline CUDA code
batchnorm2d_op = load_inline(
    name="batchnorm2d",
    cpp_sources=batchnorm2d_cpp_source,
    cuda_sources=batchnorm2d_source,
    functions=["batchnorm2d_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA kernel for BatchNorm2d.
    """

    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=num_features)
        self.batchnorm2d_cuda = batchnorm2d_op.batchnorm2d_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use the custom CUDA implementation during forward pass
        weight = self.bn.weight.data
        bias = self.bn.bias.data
        running_mean = self.bn.running_mean
        running_var = self.bn.running_var
        eps = self.bn.eps
        return self.batchnorm2d_cuda(x, weight, bias, running_mean, running_var, eps)


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Batch Normalization.
    """
    def __init__(self, num_features: int):
        """
        Initializes the BatchNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(Model, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Batch Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Batch Normalization applied, same shape as input.
        """
        return self.bn(x)

batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [features]
