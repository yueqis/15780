import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA implementation of InstanceNorm2d
instance_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void instance_norm_kernel(const float* input, float* output, 
                                    const float* weight, const float* bias,
                                    int batch_size, int channels, 
                                    int height, int width, 
                                    float eps) {
    int b = blockIdx.z;
    int c = blockIdx.y;
    int x = threadIdx.x;
    
    // Each block handles one channel and one batch
    const float* in_ptr = input + b * channels * height * width + c * height * width;
    float* out_ptr = output + b * channels * height * width + c * height * width;
    
    // Compute mean
    float sum = 0.0f;
    for (int i = 0; i < height * width; i++) {
        sum += in_ptr[i];
    }
    float mean = sum / (height * width);
    
    // Compute variance
    float var = 0.0f;
    for (int i = 0; i < height * width; i++) {
        float diff = in_ptr[i] - mean;
        var += diff * diff;
    }
    var = var / (height * width) + eps;
    
    // Normalize
    float scale = (weight != nullptr) ? weight[c] / sqrt(var) : 1.0f / sqrt(var);
    float shift = (bias != nullptr) ? bias[c] : 0.0f;
    
    for (int i = 0; i < height * width; i++) {
        out_ptr[i] = (in_ptr[i] - mean) * scale + shift;
    }
}

torch::Tensor instance_norm_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                                int64_t batch_size, int64_t channels, int64_t height, int64_t width,
                                float eps) {
    auto output = torch::empty_like(input);
    
    dim3 blocks(1, channels, batch_size);  // (x-dim, y-dim, z-dim)
    dim3 threads(height * width);
    
    if (threads.x > 1024) {
        threads.x = 1024;
        blocks.x = (height * width + threads.x - 1) / threads.x;
    }

    instance_norm_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        weight.data_ptr<float>(), bias.data_ptr<float>(),
        batch_size, channels, height, width, eps);

    return output;
}
"""

instance_norm_cpp_source = """
torch::Tensor instance_norm_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                                int64_t batch_size, int64_t channels, int64_t height, int64_t width,
                                float eps);
"""

# Compile the inline CUDA code
instance_norm_op = load_inline(
    name="instance_norm",
    cpp_sources=instance_norm_cpp_source,
    cuda_sources=instance_norm_source,
    functions=["instance_norm_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA kernel for Instance Normalization.
    """

    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.inorm = nn.InstanceNorm2d(num_features=num_features)
        self.instance_norm_op = instance_norm_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract parameters from original InstanceNorm layer
        weight = (
            self.inorm.weight if self.inorm.elementwise_affine else torch.tensor([])
        )
        bias = self.inorm.bias if self.inorm.elementwise_affine else torch.tensor([])

        batch_size, channels, height, width = x.shape
        return self.instance_norm_op.instance_norm_cuda(
            x, weight, bias, batch_size, channels, height, width, self.inorm.eps
        )


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Instance Normalization.
    """
    def __init__(self, num_features: int):
        """
        Initializes the InstanceNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(Model, self).__init__()
        self.inorm = nn.InstanceNorm2d(num_features=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Instance Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).

        Returns:
            torch.Tensor: Output tensor with Instance Normalization applied, same shape as input.
        """
        return self.inorm(x)

batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [features]
