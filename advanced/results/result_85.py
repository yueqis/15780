import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for depthwise 2D convolution
depthwise_conv2d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void depthwise_conv2d_kernel(
    const float* input, 
    const float* weight, 
    float* output,
    int batch_size, 
    int channels, 
    int height, 
    int width,
    int kernel_h, 
    int kernel_w,
    int stride_h, 
    int stride_w,
    int padding_h, 
    int padding_w,
    int dilation_h, 
    int dilation_w) {
    
    int out_w = (width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    int out_h = (height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * out_h * out_w) return;

    int b = idx / (channels * out_h * out_w);
    int ch = (idx / (out_h * out_w)) % channels;
    int h_out = (idx / out_w) % out_h;
    int w_out = idx % out_w;

    int h_in = h_out * stride_h - padding_h;
    int w_in = w_out * stride_w - padding_w;

    float acc = 0.0f;
    for (int i = 0; i < kernel_h; ++i) {
        for (int j = 0; j < kernel_w; ++j) {
            int h = h_in + dilation_h * i;
            int w = w_in + dilation_w * j;
            if ((unsigned int)h < (unsigned int)height && (unsigned int)w < (unsigned int)width) {
                acc += input[b * channels * height * width + ch * height * width + h * width + w] *
                       weight[ch * kernel_h * kernel_w + i * kernel_w + j];
            }
        }
    }
    output[idx] = acc;
}

torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input, 
    torch::Tensor weight,
    int kernel_h, 
    int kernel_w,
    int stride_h, 
    int stride_w,
    int padding_h, 
    int padding_w,
    int dilation_h, 
    int dilation_w) {
    
    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    int out_w = (width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    int out_h = (height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;

    auto output = torch::zeros({batch_size, channels, out_h, out_w}, input.options());

    const int threads_per_block = 256;
    const int blocks = (batch_size * channels * out_h * out_w + threads_per_block - 1) / threads_per_block;

    depthwise_conv2d_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        output.data_ptr<float>(),
        batch_size, channels, height, width,
        kernel_h, kernel_w,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w);

    return output;
}
"""

depthwise_conv2d_cpp_source = """
torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input, 
    torch::Tensor weight,
    int kernel_h, 
    int kernel_w,
    int stride_h, 
    int stride_w,
    int padding_h, 
    int padding_w,
    int dilation_h, 
    int dilation_w);
"""

# Compile the inline CUDA code
depthwise_conv2d_op = load_inline(
    name="depthwise_conv2d",
    cpp_sources=depthwise_conv2d_cpp_source,
    cuda_sources=depthwise_conv2d_cuda_source,
    functions=["depthwise_conv2d_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size_h: int,
        kernel_size_w: int,
        stride_h: int = 1,
        stride_w: int = 1,
        padding_h: int = 0,
        padding_w: int = 0,
        dilation_h: int = 1,
        dilation_w: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.weight = nn.Parameter(
            torch.randn(in_channels, 1, kernel_size_h, kernel_size_w)
        )
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None
        self.depthwise_conv2d_cuda = depthwise_conv2d_op.depthwise_conv2d_cuda
        self.bias_flag = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.depthwise_conv2d_cuda(
            x,
            self.weight,
            self.kernel_size_h,
            self.kernel_size_w,
            self.stride_h,
            self.stride_w,
            self.padding_h,
            self.padding_w,
            self.dilation_h,
            self.dilation_w,
        )
        if self.bias_flag:
            out += self.bias.view(1, -1, 1, 1)
        return out


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a depthwise 2D convolution with asymmetric input and asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size_h (int): Height of the convolution kernel.
        kernel_size_w (int): Width of the convolution kernel.
        stride_h (int, optional): Stride of the convolution in height dimension. Defaults to 1.
        stride_w (int, optional): Stride of the convolution in width dimension. Defaults to 1.
        padding_h (int, optional): Padding applied to the input in height dimension. Defaults to 0.
        padding_w (int, optional): Padding applied to the input in width dimension. Defaults to 0.
        dilation_h (int, optional): Spacing between kernel elements in height dimension. Defaults to 1.
        dilation_w (int, optional): Spacing between kernel elements in width dimension. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, in_channels, (kernel_size_h, kernel_size_w), stride=(stride_h, stride_w), padding=(padding_h, padding_w), dilation=(dilation_h, dilation_w), groups=in_channels, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return self.conv2d(x)

# Test code
batch_size = 16
in_channels = 3
out_channels = in_channels
kernel_size_h = 3
kernel_size_w = 5
width = 256
height = 128
stride_h = 1
stride_w = 1
padding_h = 0
padding_w = 0
dilation_h = 1
dilation_w = 1
groups = in_channels

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size_h, kernel_size_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups]
