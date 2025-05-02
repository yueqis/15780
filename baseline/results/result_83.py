import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for depthwise 2D convolution with asymmetric (k x 1) kernel
depthwise_conv2d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void depthwise_conv2d_kernel(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {

    const int h_out = blockIdx.y;
    const int w_out = blockIdx.z;
    const int channel = blockIdx.x;
    const int b = threadIdx.x;

    if (b >= batch_size) return;

    float acc = 0.0f;
    for (int k = 0; k < kernel_size; ++k) {
        int pad_input_col = w_out * stride - padding + k * dilation;
        if (pad_input_col >= 0 && pad_input_col < in_width) {
            acc += input[b * channels * in_height * in_width + channel * in_height * in_width + h_out * in_width + pad_input_col] *
                   weight[channel * kernel_size + k];
        }
    }

    output[b * channels * out_height * out_width + channel * out_height * out_width + h_out * out_width + w_out] = acc;
}

torch::Tensor depthwise_conv2d_cuda(torch::Tensor input, torch::Tensor weight,
                                   int stride, int padding, int dilation, int kernel_size) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto in_height = input.size(2);
    auto in_width = input.size(3);

    int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_height = in_height; // Since we're using (k x 1) kernel and no vertical filtering

    auto output = torch::zeros({batch_size, channels, out_height, out_width}, input.options());

    dim3 blocks(channels, out_height, out_width); // One block per channel, y-output, x-output
    dim3 threads(batch_size); // One thread per batch item

    depthwise_conv2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        in_height,
        in_width,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding,
        dilation);

    return output;
}
"""

depthwise_conv2d_cpp_source = """
torch::Tensor depthwise_conv2d_cuda(torch::Tensor input, torch::Tensor weight,
                                   int stride, int padding, int dilation, int kernel_size);
"""

# Compile the inline CUDA code
depthwise_conv2d = load_inline(
    name="depthwise_conv2d",
    cpp_sources=depthwise_conv2d_cpp_source,
    cuda_sources=depthwise_conv2d_cuda_source,
    functions=["depthwise_conv2d_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Learnable parameters
        self.weight = nn.Parameter(torch.Tensor(in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(in_channels))
        else:
            self.register_parameter("bias", None)

        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, nonlinearity="linear")
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        # Register custom op
        self.custom_op = depthwise_conv2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Call the custom CUDA implementation
        return self.custom_op.depthwise_conv2d_cuda(
            x, self.weight, self.stride, self.padding, self.dilation, self.kernel_size
        )


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a depthwise 2D convolution with a square input and an asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=(kernel_size, 1), stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, in_channels, height_out, width_out).
        """
        return self.conv2d(x)

# Test code
batch_size = 16
in_channels = 3
kernel_size = 3
width = 256
height = 256
stride = 1
padding = 0
dilation = 1

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, kernel_size, stride, padding, dilation]
