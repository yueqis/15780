import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for transposed 3D convolution
conv_transpose3d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose3d_kernel(
    const float* input, 
    const float* weight, 
    const float* bias,
    float* output,
    int batch_size, int in_channels, int out_channels,
    int depth, int height, int width,
    int kernel_size, int stride, int padding, int output_padding,
    int groups) {

    // Global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Output dimensions
    int out_depth = (depth - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_height = (height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_width = (width - 1) * stride - 2 * padding + kernel_size + output_padding;

    int elements_per_output = out_channels * out_depth * out_height * out_width;
    int total_elements = batch_size * elements_per_output;

    for (; idx < total_elements; idx += gridDim.x * blockDim.x) {
        int b = idx / elements_per_output;
        int rem = idx % elements_per_output;
        int c_out = rem / (out_depth * out_height * out_width);
        rem %= out_depth * out_height * out_width;
        int d_out = rem / (out_height * out_width);
        int h_out = (rem / out_width) % out_height;
        int w_out = rem % out_width;

        // Compute input indices based on transposed convolution logic
        float val = 0.0f;

        int group = c_out / (out_channels / groups);
        int in_ch_group = in_channels / groups;
        int start_in_ch = group * in_ch_group;

        for (int c_in = 0; c_in < in_ch_group; ++c_in) {
            for (int kd = 0; kd < kernel_size; ++kd) {
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        int d_in = d_out * stride - padding + kd;
                        int h_in = h_out * stride - padding + kh;
                        int w_in = w_out * stride - padding + kw;

                        if (d_in >= 0 && d_in < depth && h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                            int input_idx = b * in_channels * depth * height * width + (start_in_ch + c_in) * depth * height * width + d_in * height * width + h_in * width + w_in;
                            int weight_idx = c_out * in_ch_group * kernel_size * kernel_size * kernel_size + (c_in * kernel_size + kd) * kernel_size * kernel_size + kh * kernel_size + kw;
                            val += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }

        if (bias != nullptr) {
            val += bias[c_out];
        }

        int out_idx = b * out_channels * out_depth * out_height * out_width + c_out * out_depth * out_height * out_width + d_out * out_height * out_width + h_out * out_width + w_out;
        output[out_idx] = val;
    }
}

torch::Tensor conv_transpose3d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                                   int batch_size, int in_channels, int out_channels,
                                   int depth, int height, int width,
                                   int kernel_size, int stride, int padding, int output_padding,
                                   int groups) {
    int out_depth = (depth - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_height = (height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_width = (width - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output = torch::zeros({batch_size, out_channels, out_depth, out_height, out_width}, input.options());

    const int block_size = 256;
    const int num_blocks = 1024;  // Heuristic choice; can be tuned

    conv_transpose3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        bias ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        depth, height, width,
        kernel_size, stride, padding, output_padding,
        groups);

    return output;
}
"""

conv_transpose3d_cpp_source = """
torch::Tensor conv_transpose3d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                                 int batch_size, int in_channels, int out_channels,
                                 int depth, int height, int width,
                                 int kernel_size, int stride, int padding, int output_padding,
                                 int groups);
"""

# Compile the inline CUDA code
conv_transpose3d_op = load_inline(
    name="conv_transpose3d",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_cuda_source,
    functions=["conv_transpose3d_cuda"],
    verbose=False,
    with_cuda=True,
)


class ModelNew(nn.Module):
    """
    Custom implementation of transposed 3D convolution using CUDA.
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

        # Initialize weights and bias similar to PyTorch ConvTranspose3d
        k = 1 / (in_channels * kernel_size**3)
        self.weight = nn.Parameter(
            torch.Tensor(
                out_channels,
                in_channels // groups,
                kernel_size,
                kernel_size,
                kernel_size,
            ).uniform_(-k, k)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels).uniform_(-k, k))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, depth, height, width = x.size()
        return conv_transpose3d_op.conv_transpose3d_cuda(
            x,
            self.weight,
            self.bias,
            batch_size,
            self.in_channels,
            self.out_channels,
            depth,
            height,
            width,
            self.kernel_size,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
        )


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a transposed 3D convolution with square input and square kernel.

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
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size, kernel_size), stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        return self.conv_transpose3d(x)

# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
depth = 32
height = 32
width = 32

def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization
