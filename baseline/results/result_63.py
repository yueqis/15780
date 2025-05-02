import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for 2D convolution
conv2d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv2d_kernel(
    const float* input, 
    const float* weight, 
    float* output,
    int batch_size, 
    int in_channels, 
    int out_channels, 
    int input_height, 
    int input_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    int oc = blockIdx.x;  // Output channel
    int b = blockIdx.y;   // Batch index
    int oh = threadIdx.x; // Output height index
    int ow = threadIdx.y; // Output width index

    int output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int output_width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    if (oh >= output_height || ow >= output_width)
        return;

    int group_id = oc / (out_channels / groups);
    int group_in_channels = in_channels / groups;
    int group_oc = oc % (out_channels / groups);

    int weight_offset = group_id * (out_channels / groups) * group_in_channels * kernel_size * kernel_size +
                        group_oc * group_in_channels * kernel_size * kernel_size;

    float acc = 0.0f;

    for (int ic = 0; ic < group_in_channels; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int h_pad = oh * stride + kh * dilation - padding;
                int w_pad = ow * stride + kw * dilation - padding;

                if (h_pad >= 0 && h_pad < input_height && w_pad >= 0 && w_pad < input_width) {
                    int input_idx = b * in_channels * input_height * input_width +
                                    (group_id * group_in_channels + ic) * input_height * input_width +
                                    h_pad * input_width + w_pad;

                    int weight_idx = weight_offset + (ic * kernel_size + kh) * kernel_size + kw;

                    acc += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    int output_idx = b * out_channels * output_height * output_width +
                     oc * output_height * output_width +
                     oh * output_width + ow;

    output[output_idx] = acc;
}

torch::Tensor conv2d_cuda(
    torch::Tensor input, 
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    auto input_sizes = input.sizes();
    auto weight_sizes = weight.sizes();

    int batch_size = input_sizes[0];
    int in_channels = input_sizes[1];
    int input_height = input_sizes[2];
    int input_width = input_sizes[3];

    int out_channels = weight_sizes[0];
    int kernel_size = weight_sizes[2];

    int output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int output_width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, input.options());

    dim3 blocks(out_channels);
    dim3 threads(output_height, output_width);

    conv2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_size,
        stride,
        padding,
        dilation,
        groups
    );

    return output;
}
"""

conv2d_cpp_source = """
torch::Tensor conv2d_cuda(
    torch::Tensor input, 
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation,
    int groups
);
"""

# Compile the inline CUDA code
conv2d_op = load_inline(
    name="conv2d_cuda",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_cuda_source,
    functions=["conv2d_cuda"],
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
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.conv2d_cuda = conv2d_op.conv2d_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2d_cuda(
            x, self.conv.weight, self.stride, self.padding, self.dilation, self.groups
        )


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a standard 2D convolution operation with a square input and square kernel.

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
        self.conv2d = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return self.conv2d(x)

# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
width = 256
height = 256

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization
