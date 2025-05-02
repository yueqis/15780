import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for 3D convolution
conv3d_cuda_code = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv3d_kernel(
    const float* input, 
    const float* weight, 
    float* output,
    int batch_size, 
    int in_channels, 
    int out_channels,
    int input_depth, 
    int input_width, 
    int input_height,
    int kernel_size,
    int stride, 
    int padding,
    int dilation,
    int groups,
    int out_depth, 
    int out_width, 
    int out_height) {
    
    int oc_per_group = out_channels / groups;
    int ic_per_group = in_channels / groups;

    int g = blockIdx.z;
    int oc = g * oc_per_group + threadIdx.z;
    int b = blockIdx.w;
    int oz = blockIdx.x;
    int oy = blockIdx.y;
    int ox = threadIdx.y;
    int oz_start = oz * stride - padding;
    int oy_start = oy * stride - padding;
    int ox_start = ox * stride - padding;

    int input_offset = ((b * in_channels + g * ic_per_group) * input_depth * input_width * input_height);
    int weight_offset = (g * oc_per_group * ic_per_group + oc % oc_per_group) * kernel_size * kernel_size * kernel_size * ic_per_group;

    float sum = 0.0f;

    for (int kz = 0; kz < kernel_size; ++kz) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int iz = oz_start + kz * dilation;
                int iy = oy_start + ky * dilation;
                int ix = ox_start + kx * dilation;

                if (iz >= 0 && iz < input_depth &&
                    iy >= 0 && iy < input_width &&
                    ix >= 0 && ix < input_height) {
                    
                    for (int ic = 0; ic < ic_per_group; ++ic) {
                        float input_val = input[input_offset + (ic_per_group * (iz * input_width * input_height + iy * input_height + ix)) + ic];
                        float weight_val = weight[weight_offset + (ic * kernel_size * kernel_size * kernel_size + kz * kernel_size * kernel_size + ky * kernel_size + kx)];
                        sum += input_val * weight_val;
                    }
                }
            }
        }
    }

    output[b * out_channels * out_depth * out_width * out_height +
           oc * out_depth * out_width * out_height +
           oz * out_width * out_height +
           oy * out_height +
           ox] = sum;
}

torch::Tensor custom_conv3d_cuda(torch::Tensor input, torch::Tensor weight,
                                 int kernel_size, int stride, int padding, int dilation,
                                 int groups) {
    auto input_sizes = input.sizes();
    int batch_size = input_sizes[0];
    int in_channels = input_sizes[1];
    int input_depth = input_sizes[2];
    int input_width = input_sizes[3];
    int input_height = input_sizes[4];

    int out_channels = weight.size(0); // Should be divisible by groups
    int out_depth = (input_depth + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, out_depth, out_width, out_height}, input.options());

    dim3 blocks(out_depth, out_width, groups);
    dim3 threads(8, 8, 8); // Tunable based on architecture

    conv3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_depth, input_width, input_height,
        kernel_size, stride, padding, dilation,
        groups, out_depth, out_width, out_height);

    return output;
}
"""

conv3d_cpp_code = """
torch::Tensor custom_conv3d_cuda(torch::Tensor input, torch::Tensor weight,
                                 int kernel_size, int stride, int padding, int dilation,
                                 int groups);
"""

# Compile the inline CUDA code
custom_conv3d = load_inline(
    name="custom_conv3d",
    cpp_sources=conv3d_cpp_code,
    cuda_sources=conv3d_cuda_code,
    functions=["custom_conv3d_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(
            torch.Tensor(
                out_channels,
                in_channels // groups,
                kernel_size,
                kernel_size,
                kernel_size,
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.reset_parameters()

        # Load our custom CUDA operator
        self.custom_conv3d_op = custom_conv3d

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, mode="fan_out", nonlinearity="relu")
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.custom_conv3d_op.custom_conv3d_cuda(
            x,
            self.weight,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1, 1)

        return output


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a standard 3D convolution operation with square input and square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, (kernel_size, kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, width, height).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, width_out, height_out).
        """
        return self.conv3d(x)

# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
depth = 64
width = 64
height = 64

def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, width, height)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization
