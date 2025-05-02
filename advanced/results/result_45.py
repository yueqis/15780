import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 2D Average Pooling
avg_pool_2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool_2d_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding
) {
    int n = blockIdx.z;
    int c = blockIdx.y;
    int out_h = blockIdx.x / (gridDim.y * gridDim.z);
    int out_w = blockIdx.x % (gridDim.y * gridDim.z);
    out_h = out_h * blockDim.x + threadIdx.x;
    out_w = out_w * blockDim.y + threadIdx.y;

    if (out_h >= (height + 2 * padding - kernel_size) / stride + 1 ||
        out_w >= (width + 2 * padding - kernel_size) / stride + 1) {
        return;
    }

    int in_h_start = out_h * stride - padding;
    int in_w_start = out_w * stride - padding;
    int sum_count = 0;
    float sum = 0.0f;

    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int h = in_h_start + kh;
            int w = in_w_start + kw;
            if (h >= 0 && h < height && w >= 0 && w < width) {
                sum += input[(n * channels + c) * height * width + h * width + w];
                sum_count++;
            }
        }
    }

    if (sum_count > 0) {
        output[(n * channels + c) * ((height + 2 * padding - kernel_size) / stride + 1) *
               ((width + 2 * padding - kernel_size) / stride + 1) +
               out_h * ((width + 2 * padding - kernel_size) / stride + 1) + out_w] =
            sum / sum_count;
    }
}

torch::Tensor avg_pool_2d_cuda(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding
) {
    auto input_sizes = input.sizes();
    int batch_size = input_sizes[0];
    int channels = input_sizes[1];
    int height = input_sizes[2];
    int width = input_sizes[3];

    int out_h = (height + 2 * padding - kernel_size) / stride + 1;
    int out_w = (width + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::zeros({batch_size, channels, out_h, out_w}, input.options());

    dim3 block_dim(16, 16);
    dim3 grid_dim(
        ((out_h * out_w) + (block_dim.x * block_dim.y) - 1) / (block_dim.x * block_dim.y),
        channels,
        batch_size
    );

    avg_pool_2d_kernel<<<grid_dim, block_dim>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width,
        kernel_size,
        stride,
        padding
    );

    return output;
}
"""

avg_pool_2d_cpp_source = """
torch::Tensor avg_pool_2d_cuda(torch::Tensor input, int kernel_size, int stride, int padding);
"""

# Compile the inline CUDA code
avg_pool_2d_op = load_inline(
    name="avg_pool_2d",
    cpp_sources=avg_pool_2d_cpp_source,
    cuda_sources=avg_pool_2d_source,
    functions=["avg_pool_2d_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model that performs 2D Average Pooling using a custom CUDA kernel.
    """

    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer using a custom CUDA implementation.
        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to None (same as kernel_size).
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.avg_pool_cuda = avg_pool_2d_op.avg_pool_2d_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies 2D Average Pooling using a custom CUDA kernel.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        return self.avg_pool_cuda(x, self.kernel_size, self.stride, self.padding)


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs 2D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to None (same as kernel_size).
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(Model, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies 2D Average Pooling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        return self.avg_pool(x)

batch_size = 16
channels = 64
height = 256
width = 256
kernel_size = 3

def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    return [kernel_size]
