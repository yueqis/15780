import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Conv2d with asymmetric kernel, padding, dilation
conv2d_custom_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv2d_custom_kernel(
    const float* input, 
    const float* weight, 
    float* output,
    int batch_size, 
    int in_channels, 
    int out_channels,
    int input_height, 
    int input_width,
    int output_height, 
    int output_width,
    int kernel_h, 
    int kernel_w,
    int stride_h, 
    int stride_w,
    int pad_top, 
    int pad_bottom, 
    int pad_left, 
    int pad_right,
    int dilation_h, 
    int dilation_w) {

    int n = blockIdx.z;
    int oc = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int w_out = input_width + pad_left + pad_right;
    int h_out = input_height + pad_top + pad_bottom;

    int px = idx % output_width;
    int py = idx / output_width;
    
    if (px >= output_width || py >= output_height) return;

    // Calculate the starting position in padded input
    int start_x = px * stride_w - pad_left;
    int start_y = py * stride_h - pad_top;

    float acc = 0.0f;

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int iy = start_y + kh * dilation_h;
                int ix = start_x + kw * dilation_w;

                if (iy >= 0 && iy < input_height && ix >= 0 && ix < input_width) {
                    float in_val = input[n * in_channels * input_height * input_width +
                                         ic * input_height * input_width +
                                         iy * input_width + ix];

                    float w_val = weight[oc * in_channels * kernel_h * kernel_w +
                                         ic * kernel_h * kernel_w +
                                         kh * kernel_w + kw];

                    acc += in_val * w_val;
                }
            }
        }
    }

    output[n * out_channels * output_height * output_width +
           oc * output_height * output_width +
           py * output_width + px] = acc;
}

torch::Tensor conv2d_custom_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_h,
    int stride_w,
    int pad_top,
    int pad_bottom,
    int pad_left,
    int pad_right,
    int dilation_h,
    int dilation_w,
    int kernel_h,
    int kernel_w) {

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);

    int out_channels = weight.size(0);

    int output_height = (input_height + pad_top + pad_bottom - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int output_width = (input_width + pad_left + pad_right - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, input.options());

    dim3 blocks(output_width * output_height);
    dim3 grids(1, out_channels, batch_size);

    const int block_size = 256;
    const int num_blocks = (output_width * output_height + block_size - 1) / block_size;

    conv2d_custom_kernel<<<grids, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_height, input_width,
        output_height, output_width,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_top, pad_bottom, pad_left, pad_right,
        dilation_h, dilation_w);

    return output;
}
"""

conv2d_custom_cpp_source = """
torch::Tensor conv2d_custom_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_h,
    int stride_w,
    int pad_top,
    int pad_bottom,
    int pad_left,
    int pad_right,
    int dilation_h,
    int dilation_w,
    int kernel_h,
    int kernel_w);
"""

# Compile the inline CUDA code
conv2d_custom_op = load_inline(
    name="conv2d_custom",
    cpp_sources=conv2d_custom_cpp_source,
    cuda_sources=conv2d_custom_cuda_source,
    functions=["conv2d_custom_cuda"],
    verbose=False,
    extra_cflags=["-O2"],
    with_cuda=True,
)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: int = 1,
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size[0], kernel_size[1])
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        # Load custom CUDA op
        self.conv2d_custom = conv2d_custom_op

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity="linear")
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply custom CUDA convolution
        out = self.conv2d_custom.conv2d_custom_cuda(
            x,
            self.weight,
            self.stride,
            self.stride,
            self.padding[0],
            self.padding[0],  # Assuming symmetric padding top/bottom
            self.padding[1],
            self.padding[1],  # Assuming symmetric padding left/right
            self.dilation[0],
            self.dilation[1],
            self.kernel_size[0],
            self.kernel_size[1],
        )

        if self.bias is not None:
            out += self.bias.view(1, -1, 1, 1)

        return out


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a standard 2D convolution operation with square input and asymmetric kernel, with dilation and padding.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (height, width). 
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (tuple, optional): Padding applied to the input (top/bottom, left/right). Defaults to (0, 0).
        dilation (tuple, optional): Spacing between kernel elements (height, width). Defaults to (1, 1).
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1), bias: bool = False):
        super(Model, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        
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
kernel_size = (3, 5) # Asymmetric kernel
width = 256
height = 256
stride = 1
padding = (1, 2) # Asymmetric padding
dilation = (2, 1) # Asymmetric dilation

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]
