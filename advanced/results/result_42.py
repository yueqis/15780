import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for Max Pooling 2D
maxpool2d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define DIVUP(a, b) ((a + b - 1) / b)

__global__ void maxpool2d_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int pooled_h,
    int pooled_w
) {
    int n = blockIdx.z;
    int c = blockIdx.y;
    int out_y = blockIdx.x;
    int out_x = threadIdx.x;

    int out_idx = out_y * blockDim.x + out_x;
    if (out_idx >= pooled_h * pooled_w) return;

    int ox = out_idx % pooled_w;
    int oy = out_idx / pooled_w;

    int ix_start = ox * stride_w - pad_w;
    int iy_start = oy * stride_h - pad_h;

    float max_val = -1e9f;

    for (int ky = 0; ky < kernel_h; ++ky) {
        for (int kx = 0; kx < kernel_w; ++kx) {
            int iy = iy_start + ky * dilation_h;
            int ix = ix_start + kx * dilation_w;

            if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
                int in_idx = n * channels * height * width +
                             c * height * width +
                             iy * width + ix;
                max_val = fmaxf(max_val, input[in_idx]);
            }
        }
    }

    int out_linear_idx = n * channels * pooled_h * pooled_w +
                         c * pooled_h * pooled_w +
                         oy * pooled_w + ox;
    output[out_linear_idx] = max_val;
}

torch::Tensor maxpool2d_cuda(
    torch::Tensor input,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w
) {
    auto options = input.options();
    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    int pooled_h = (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int pooled_w = (width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    torch::Tensor output = torch::zeros({batch_size, channels, pooled_h, pooled_w}, options);

    dim3 blocks(1, channels, batch_size); // Each channel and batch processed independently
    int threads = pooled_h * pooled_w;
    if (threads > 1024) threads = 1024;

    maxpool2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        pooled_h,
        pooled_w
    );

    return output;
}
"""

# C++ wrapper for the CUDA kernel
maxpool2d_cpp_source = """
torch::Tensor maxpool2d_cuda(
    torch::Tensor input,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w
);
"""

# Compile the inline CUDA code
maxpool2d_op = load_inline(
    name="maxpool2d",
    cpp_sources=maxpool2d_cpp_source,
    cuda_sources=maxpool2d_cuda_source,
    functions=["maxpool2d_cuda"],
    verbose=True,
    with_cuda=True,
)


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.maxpool_cuda = maxpool2d_op.maxpool2d_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is on the GPU
        x = x.cuda()
        return self.maxpool_cuda(
            x,
            self.kernel_size,
            self.kernel_size,
            self.stride,
            self.stride,
            self.padding,
            self.padding,
            self.dilation,
            self.dilation,
        )


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Max Pooling 2D.
    """
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        """
        Initializes the Max Pooling 2D layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int): Stride of the pooling window.
            padding (int): Padding to be applied before pooling.
            dilation (int): Spacing between kernel elements.
        """
        super(Model, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 2D to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after Max Pooling 2D, shape (batch_size, channels, pooled_height, pooled_width).
        """
        return self.maxpool(x)

batch_size = 16
channels = 32
height = 128
width = 128
kernel_size = 2
stride = 2
padding = 1
dilation = 3

def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation]

