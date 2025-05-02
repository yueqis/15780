```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused BatchNorm2d operation
batchnorm_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void batchnorm_forward_kernel(
    const float* input, 
    float* output,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float momentum,
    float eps,
    int N, int C, int H, int W) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < N * C * H * W; i += stride) {
        int n = i / (C * H * W);
        int c = (i / (H * W)) % C;
        int h = (i / W) % H;
        int w = i % W;

        float x = input[i];
        float mean = running_mean[c];
        float var = running_var[c];
        float scale = weight[c] / sqrt(var + eps);
        float shift = bias[c] - mean * scale;

        output[i] = x * scale + shift;
    }
}

torch::Tensor batchnorm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double momentum,
    double eps) {

    auto output = torch::empty_like(input);
    auto N = input.size(0);
    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);

    const int block_size = 256;
    const int num_blocks = 1024;

    batchnorm_forward_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        momentum,
        eps,
        N, C, H, W);

    return output;
}
"""

batchnorm_cpp_source = """
torch::Tensor batchnorm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double momentum,
    double eps);
"""

# Compile the inline CUDA code
batchnorm_op = load_inline(
    name="batchnorm",
    cpp_sources=batchnorm_cpp_source,
    cuda_sources=batchnorm_cuda_source,
    functions=["batchnorm_forward_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA kernel for BatchNorm2d.
    """
    def __init__(self, num_features: int):
        """
        Initializes the BatchNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(ModelNew, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=num_features)
        self.batchnorm_forward_cuda = batchnorm_op.batchnorm_forward_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies custom CUDA implementation of Batch Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).
            
        Returns:
            torch.Tensor: Output tensor with Batch Normalization applied, same shape as input.
        """
        return self.batchnorm_forward_cuda(
            x, 
            self.bn.weight, 
            self.bn.bias, 
            self.bn.running_mean, 
            self.bn.running_var, 
            self.bn.momentum, 
            self.bn.eps
        )
```