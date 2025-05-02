import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Conv2d with asymmetric input and square kernel
custom_conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void conv2d_kernel(const float* input, const float* weight, float* output,
                              int batch_size, int in_channels, int out_channels,
                              int height, int width, int kernel_size,
                              int stride, int padding, int dilation) {
    int oc = blockIdx.x;
    int b = blockIdx.y;
    int x = threadIdx.x;
    int y = threadIdx.y;

    extern __shared__ float shared_mem[];

    // Shared memory allocation for input tile
    float* input_tile = shared_mem;
    int tile_size = (BLOCK_SIZE + kernel_size - 1) * (BLOCK_SIZE + kernel_size - 1);
    float* weight_tile = input_tile + tile_size;

    // Compute output dimensions
    int out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    // Calculate global output position
    int oy = y + blockIdx.z * BLOCK_SIZE;
    int ox = x + (blockIdx.z % (out_width / BLOCK_SIZE)) * BLOCK_SIZE;

    if (oy >= out_height || ox >= out_width)
        return;

    // Load weights into shared memory
    for (int i = threadIdx.x; i < kernel_size * kernel_size * in_channels; i += blockDim.x) {
        weight_tile[i] = weight[oc * kernel_size * kernel_size * in_channels + i];
    }
    __syncthreads();

    // Initialize output value
    float acc = 0.0f;

    // Perform convolution
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                // Calculate input coordinates with dilation and padding
                int iy = oy * stride + ky * dilation - padding;
                int ix = ox * stride + kx * dilation - padding;

                // Check bounds and load input value
                float val = (iy >= 0 && iy < height && ix >= 0 && ix < width)
                            ? input[((b * in_channels + ic) * height + iy) * width + ix]
                            : 0.0f;

                // Multiply with corresponding weight and accumulate
                acc += val * weight_tile[(ic * kernel_size * kernel_size + ky * kernel_size + kx)];
            }
        }
    }

    // Store result in output tensor
    output[(b * out_channels + oc) * out_height * out_width + oy * out_width + ox] = acc;
}

torch::Tensor custom_conv2d_cuda(torch::Tensor input, torch::Tensor weight,
                                int kernel_size, int stride, int padding, int dilation) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_channels = weight.size(0);

    // Compute output dimensions
    int out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    // Create output tensor
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

    // Set up grid and block dimensions
    dim3 grid(out_channels, batch_size, (out_height * out_width) / (BLOCK_SIZE * BLOCK_SIZE) + 1);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    // Calculate shared memory size
    int tile_size = (BLOCK_SIZE + kernel_size - 1) * (BLOCK_SIZE + kernel_size - 1);
    int smem_size = tile_size * sizeof(float) + kernel_size * kernel_size * in_channels * sizeof(float);

    // Launch kernel
    conv2d_kernel<<<grid, block, smem_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), 
                                             output.data_ptr<float>(),
                                             batch_size, in_channels, out_channels, 
                                             height, width, kernel_size,
                                             stride, padding, dilation);

    return output;
}
"""

custom_conv2d_cpp_source = """
torch::Tensor custom_conv2d_cuda(torch::Tensor input, torch::Tensor weight,
                                int kernel_size, int stride, int padding, int dilation);
"""

# Compile the inline CUDA code
custom_conv2d = load_inline(
    name="custom_conv2d",
    cpp_sources=custom_conv2d_cpp_source,
    cuda_sources=custom_conv2d_source,
    functions=["custom_conv2d_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized 2D convolution operation with an asymmetric input and a square kernel using a custom CUDA kernel.
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
        assert groups == 1, "Groups > 1 not supported in custom convolution"
        assert not bias, "Bias not supported in custom convolution"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Register weight as a parameter
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        )
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")

        # Register the custom convolution function
        self.custom_conv2d = custom_conv2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 2D convolution.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return self.custom_conv2d.custom_conv2d_cuda(
            x, self.weight, self.kernel_size, self.stride, self.padding, self.dilation
        )


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a standard 2D convolution operation with an asymmetric input and a square kernel.

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
height = 128  # Asymmetric input

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization
