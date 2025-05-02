import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D Average Pooling
avg_pool_3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define DIVUP(a, b) ((a + b - 1) / b)

__global__ void avg_pool_3d_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int in_depth,
    int in_height,
    int in_width,
    int out_depth,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.z * blockDim.z + threadIdx.z;

    if (n >= batch_size || c >= channels || d >= out_depth || c >= out_channels || h >= out_height || w >= out_width) {
        return;
    }

    int d_start = d * stride - padding;
    int h_start = h * stride - padding;
    int w_start = w * stride - padding;

    int count = 0;
    float sum = 0.0f;

    for (int kd = 0; kd < kernel_size; ++kd) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int cur_d = d_start + kd;
                int cur_h = h_start + kh;
                int cur_w = w_start + kw;

                if (cur_d >= 0 && cur_d < in_depth &&
                    cur_h >= 0 && cur_h < in_height &&
                    cur_w >= 0 && cur_w < in_width) {
                    
                    int idx = n * channels * in_depth * in_height * in_width +
                              c * in_depth * in_height * in_width +
                              cur_d * in_height * in_width +
                              cur_h * in_width +
                              cur_w;
                    sum += input[idx];
                    count++;
                }
            }
        }
    }

    if (count > 0) {
        output[n * channels * out_depth * out_height * out_width +
               c * out_depth * out_height * out_width +
               d * out_height * out_width +
               h * out_width +
               w] = sum / count;
    } else {
        output[n * channels * out_depth * out_height * out_width +
               c * out_depth * out_height * out_width +
               d * out_height * out_width +
               h * out_width +
               w] = 0.0f;
    }
}

torch::Tensor avg_pool_3d_cuda(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding)
{
    auto options = input.options();
    int batch_size = input.size(0);
    int channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);

    int out_depth = (in_depth + 2 * padding - kernel_size) / stride + 1;
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;

    torch::Tensor output = torch::zeros({batch_size, channels, out_depth, out_height, out_width}, options);

    dim3 threads(8, 8, 4);
    dim3 blocks(
        DIVUP(batch_size, threads.x),
        DIVUP(channels, threads.y),
        DIVUP(out_depth * out_height * out_width, threads.z)
    );

    avg_pool_3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        in_depth,
        in_height,
        in_width,
        out_depth,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding);

    return output;
}
"""

avg_pool_3d_cpp_source = """
torch::Tensor avg_pool_3d_cuda(torch::Tensor input, int kernel_size, int stride, int padding);
"""

# Compile the inline CUDA code for 3D average pooling
avg_pool_3d_op = load_inline(
    name="avg_pool_3d",
    cpp_sources=avg_pool_3d_cpp_source,
    cuda_sources=avg_pool_3d_source,
    functions=["avg_pool_3d_cuda"],
    verbose=True,
    extra_cflags=["-O2"],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs 3D Average Pooling using a custom CUDA kernel.
    """

    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer with custom CUDA implementation.
        Args:
            kernel_size (int): Size of the kernel to apply pooling.
            stride (int, optional): Stride of the pooling operation. Defaults to None, which uses the kernel size.
            padding (int, optional): Padding to apply before pooling. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.avg_pool_cuda = avg_pool_3d_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies 3D Average Pooling using custom CUDA kernel.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width).
        Returns:
            torch.Tensor: Output tensor with Average Pooling applied, shape depends on kernel_size, stride and padding.
        """
        return self.avg_pool_cuda.avg_pool_3d_cuda(
            x, self.kernel_size, self.stride, self.padding
        )
