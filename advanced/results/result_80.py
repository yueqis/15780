import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 2D convolution
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
    int output_height,
    int output_width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w) {
    
    const int oh = blockIdx.x;
    const int ow = threadIdx.x;
    const int b = blockIdx.y;
    const int oc = blockIdx.z;

    // Compute output coordinates
    if (oh >= output_height || ow >= output_width) return;

    // Initialize output to zero
    float acc = 0.0f;

    // Loop over input channels
    for (int ic = 0; ic < in_channels; ++ic) {
        // Loop over kernel dimensions
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                // Compute input coordinates with dilation and padding
                const int ih = oh * stride_h + kh * dilation_h - pad_h;
                const int iw = ow * stride_w + kw * dilation_w - pad_w;

                // Check bounds
                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                    // Load input and weight
                    float input_val = input[b * in_channels * input_height * input_width + ic * input_height * input_width + ih * input_width + iw];
                    
                    float weight_val = weight[oc * in_channels * kernel_h * kernel_w + ic * kernel_h * kernel_w + kh * kernel_w + kw];
                    
                    // Accumulate result
                    acc += input_val * weight_val;
                }
            }
        }
    }

    // Write output
    output[b * out_channels * output_height * output_width + oc * output_height * output_width + oh * output_width + ow] = acc;
}

torch::Tensor conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width) {
    
    // Create output tensor
    auto output = torch::zeros({input.size(0), out_channels, output_height, output_width}, input.options());

    // Launch kernel
    dim3 blocks(output_height, output_width);
    dim3 grids(out_channels, input.size(0));

    conv2d_kernel<<<grids, blocks>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        input.size(0),
        in_channels,
        out_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w);

    return output;
}
"""

conv2d_cpp_source = """
torch::Tensor conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width);
"""

# Compile the inline CUDA code
conv2d_op = load_inline(
    name="conv2d_op",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_cuda_source,
    functions=["conv2d_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized 2D convolution operation with custom CUDA kernel.
    """

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
        self.kernel_h, self.kernel_w = kernel_size
        self.stride_h = self.stride_w = stride
        self.pad_h, self.pad_w = padding
        self.dilation_h, self.dilation_w = dilation
        self.use_bias = bias

        # Register parameters
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, self.kernel_h, self.kernel_w)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        if bias:
            nn.init.zeros_(self.bias)

        # Register the custom CUDA op
        self.conv2d_cuda = conv2d_op.conv2d_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 2D convolution.
        """
        batch_size, _, input_height, input_width = x.shape

        # Calculate output dimensions
        output_height = (
            input_height + 2 * self.pad_h - self.dilation_h * (self.kernel_h - 1) - 1
        ) // self.stride_h + 1
        output_width = (
            input_width + 2 * self.pad_w - self.dilation_w * (self.kernel_w - 1) - 1
        ) // self.stride_w + 1

        return self.conv2d_cuda(
            x,
            self.weight,
            self.bias if self.use_bias else torch.tensor([]),
            self.stride_h,
            self.stride_w,
            self.pad_h,
            self.pad_w,
            self.dilation_h,
            self.dilation_w,
            self.in_channels,
            self.out_channels,
            self.kernel_h,
            self.kernel_w,
            input_height,
            input_width,
            output_height,
            output_width,
        )


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
