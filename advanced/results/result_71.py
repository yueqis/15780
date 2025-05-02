import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for transposed 2D convolution
conv_transpose2d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose2d_kernel(
    const float* input, float* output,
    const float* weight, const float* bias,
    int batch_size, int in_channels, int out_channels,
    int height_in, int width_in,
    int kernel_size, int stride, int padding, int output_padding) {

    // Simplified transposed convolution logic (basic implementation)
    int n = blockIdx.z;
    int oh = blockIdx.y;
    int ow = blockIdx.x;
    int c_out = threadIdx.y;
    int c_in = threadIdx.x;

    float sum = 0.0f;
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int ih = oh * stride - 2 * padding + kh;
            int iw = ow * stride - 2 * padding + kw;
            if (ih >= 0 && ih < height_in && iw >= 0 && iw < width_in) {
                int input_idx = ((n * in_channels + c_in) * height_in * width_in) + ih * width_in + iw;
                int weight_idx = ((c_out * in_channels + c_in) * kernel_size * kernel_size) + kh * kernel_size + kw;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }

    int output_idx = ((n * out_channels + c_out) * (height_in * stride + output_padding) * (width_in * stride + output_padding)) +
                     oh * (width_in * stride + output_padding) + ow;
    output[output_idx] = bias ? sum + bias[c_out] : sum;
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int64_t kernel_size, int64_t stride, int64_t padding, int64_t output_padding) {

    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height_in = input.size(2);
    auto width_in = input.size(3);

    auto out_channels = weight.size(0);
    auto height_out = (height_in - 1) * stride + kernel_size - 2 * padding + output_padding;
    auto width_out = (width_in - 1) * stride + kernel_size - 2 * padding + output_padding;

    auto output = torch::zeros({batch_size, out_channels, height_out, width_out}, input.options());

    dim3 blocks(width_out, height_out, batch_size);
    dim3 threads(in_channels, out_channels, 1);

    conv_transpose2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        weight.data_ptr<float>(), bias ? bias.data_ptr<float>() : nullptr,
        batch_size, in_channels, out_channels,
        height_in, width_in,
        kernel_size, stride, padding, output_padding);

    return output;
}
"""

conv_transpose2d_cpp_source = """
torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int64_t kernel_size, int64_t stride, int64_t padding, int64_t output_padding);
"""

# Compile the inline CUDA code for the custom transposed convolution
conv_transpose2d_op = load_inline(
    name="conv_transpose2d_op",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_cuda_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized version of Model using a custom CUDA kernel for transposed 2D convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.use_bias = bias

        # Register weights and bias as parameters
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter("bias", None)

        # Register the custom CUDA op
        self.conv_transpose2d_cuda = conv_transpose2d_op.conv_transpose2d_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution using a custom CUDA kernel.
        """
        return self.conv_transpose2d_cuda(
            x,
            self.weight,
            self.bias,
            self.kernel_size,
            self.stride,
            self.padding,
            self.output_padding,
        )


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a transposed 2D convolution with asymmetric input and a square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        output_padding (int, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return self.conv_transpose2d(x)

# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
height_in = 128
width_in = 256

def get_inputs():
    x = torch.randn(batch_size, in_channels, height_in, width_in)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization
