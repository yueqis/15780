import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for transposed 3D convolution
conv_transpose3d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose3d_kernel(
    const float* input, 
    const float* weight, 
    const float* bias,
    float* output,
    int batch_size, 
    int in_channels, 
    int out_channels, 
    int depth, 
    int width, 
    int height,
    int kernel_depth,
    int kernel_width,
    int kernel_height,
    int stride_depth,
    int stride_width,
    int stride_height,
    int padding_depth,
    int padding_width,
    int padding_height,
    int output_padding_depth,
    int output_padding_width,
    int output_padding_height,
    int groups) {

    // Simple implementation of transposed convolution (no optimization for clarity and brevity)
    int oc = blockIdx.z;
    int b = blockIdx.y;
    int od = threadIdx.z;
    int ow = threadIdx.x;
    int oh = threadIdx.y;

    int odepth = depth * stride_depth - 2 * padding_depth + kernel_depth + output_padding_depth;
    int owidth = width * stride_width - 2 * padding_width + kernel_width + output_padding_width;
    int oheight = height * stride_height - 2 * padding_height + kernel_height + output_padding_height;

    float val = 0.0f;
    int group_offset = in_channels / groups * oc;

    for (int ic = 0; ic < in_channels / groups; ++ic) {
        for (int kd = 0; kd < kernel_depth; ++kd) {
            for (int kw = 0; kw < kernel_width; ++kw) {
                for (int kh = 0; kh < kernel_height; ++kh) {
                    int id = od * stride_depth - padding_depth + kd;
                    int iw = ow * stride_width - padding_width + kw;
                    int ih = oh * stride_height - padding_height + kh;

                    if (id >= 0 && id < depth && iw >= 0 && iw < width && ih >= 0 && ih < height) {
                        val += input[b * in_channels * depth * width * height + (group_offset + ic) * depth * width * height + id * width * height + iw * height + ih] *
                               weight[oc * in_channels * kernel_depth * kernel_width * kernel_height + (group_offset + ic) * kernel_depth * kernel_width * kernel_height + kd * kernel_width * kernel_height + kw * kernel_height + kh];
                    }
                }
            }
        }
    }

    if (bias != nullptr) {
        val += bias[oc];
    }

    output[b * out_channels * odepth * owidth * oheight + oc * odepth * owidth * oheight + od * owidth * oheight + ow * oheight + oh] = val;
}

torch::Tensor conv_transpose3d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
                                   int64_t stride_depth, int64_t stride_width, int64_t stride_height,
                                   int64_t padding_depth, int64_t padding_width, int64_t padding_height,
                                   int64_t output_padding_depth, int64_t output_padding_width, int64_t output_padding_height,
                                   int64_t groups) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto depth = input.size(2);
    auto width = input.size(3);
    auto height = input.size(4);

    auto out_channels = weight.size(0);
    auto kernel_depth = weight.size(2);
    auto kernel_width = weight.size(3);
    auto kernel_height = weight.size(4);

    auto odepth = depth * stride_depth - 2 * padding_depth + kernel_depth + output_padding_depth;
    auto owidth = width * stride_width - 2 * padding_width + kernel_width + output_padding_width;
    auto oheight = height * stride_height - 2 * padding_height + kernel_height + output_padding_height;

    dim3 blocks(batch_size, out_channels, 1);
    dim3 threads(odepth, owidth, oheight);

    conv_transpose3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels, depth, width, height,
        kernel_depth, kernel_width, kernel_height,
        stride_depth, stride_width, stride_height,
        padding_depth, padding_width, padding_height,
        output_padding_depth, output_padding_width, output_padding_height,
        groups);

    return output;
}
"""

conv_transpose3d_cpp_source = """
torch::Tensor conv_transpose3d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
                                   int64_t stride_depth, int64_t stride_width, int64_t stride_height,
                                   int64_t padding_depth, int64_t padding_width, int64_t padding_height,
                                   int64_t output_padding_depth, int64_t output_padding_width, int64_t output_padding_height,
                                   int64_t groups);
"""

# Compile the inline CUDA code for transposed 3D convolution
conv_transpose3d_op = load_inline(
    name="conv_transpose3d",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_cuda_source,
    functions=["conv_transpose3d_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1, 1),
        padding: tuple = (0, 0, 0),
        output_padding: tuple = (0, 0, 0),
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

        # Create weight parameter
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *kernel_size)
        )

        # Initialize weights using Kaiming uniform initialization
        nn.init.kaiming_uniform_(
            self.weight, a=0.25, mode="fan_in", nonlinearity="leaky_relu"
        )

        # Create bias parameter if needed
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Allocate output tensor
        output = torch.empty(
            x.size(0),
            self.out_channels,
            x.size(2) * self.stride[0]
            - 2 * self.padding[0]
            + self.kernel_size[0]
            + self.output_padding[0],
            x.size(3) * self.stride[1]
            - 2 * self.padding[1]
            + self.kernel_size[1]
            + self.output_padding[1],
            x.size(4) * self.stride[2]
            - 2 * self.padding[2]
            + self.kernel_size[2]
            + self.output_padding[2],
            device=x.device,
            dtype=x.dtype,
        )

        return conv_transpose3d_op.conv_transpose3d_cuda(
            x.contiguous(),
            self.weight.contiguous(),
            self.bias.contiguous() if self.bias is not None else self.bias,
            output,
            self.stride[0],
            self.stride[1],
            self.stride[2],
            self.padding[0],
            self.padding[1],
            self.padding[2],
            self.output_padding[0],
            self.output_padding[1],
            self.output_padding[2],
            self.groups,
        )


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a transposed 3D convolution with a square input and an asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (kernel_depth, kernel_width, kernel_height), 
                             where kernel_width == kernel_height.
        stride (tuple, optional): Stride of the convolution. Defaults to (1, 1, 1).
        padding (tuple, optional): Padding applied to the input. Defaults to (0, 0, 0).
        output_padding (tuple, optional): Additional size added to one side of the output shape. Defaults to (0, 0, 0).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, width, height).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, width_out, height_out).
        """
        return self.conv_transpose3d(x)

# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_depth = 3
kernel_width = 5
kernel_height = 5
depth = 64
width = 64
height = 64

def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, width, height)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, (kernel_depth, kernel_width, kernel_height)]  # Provide in_channels, out_channels, kernel_size for initialization
