import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for transposed 2D convolution
conv_transpose2d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Simplified transposed convolution kernel for square kernel size and stride
__global__ void conv_transpose2d_kernel(
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
    int output_padding,
    int groups) {

    int ox = threadIdx.x + blockIdx.x * blockDim.x;
    int oy = threadIdx.y + blockIdx.y * blockDim.y;
    int oz = threadIdx.z + blockIdx.z * blockDim.z;

    int output_height = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int output_width = output_height;

    if (ox >= output_width || oy >= output_height || oz >= batch_size * out_channels) return;

    int b = oz / out_channels;
    int oc = oz % out_channels;

    float acc = 0.0f;

    // Assuming group == 1 for this implementation
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int iy = (oy + padding - ky + stride - 1) / stride;
                int ix = (ox + padding - kx + stride - 1) / stride;

                if (iy >= 0 && iy < input_height && ix >= 0 && ix < input_width &&
                    (oy + padding - ky) % stride == 0 &&
                    (ox + padding - kx) % stride == 0) {
                    
                    float inputValue = input[b * in_channels * input_height * input_width +
                                            ic * input_height * input_width +
                                            iy * input_width + ix];
                    
                    float weightValue = weight[oc * in_channels * kernel_size * kernel_size +
                                              ic * kernel_size * kernel_size +
                                              ky * kernel_size + kx];
                    
                    acc += inputValue * weightValue;
                }
            }
        }
    }

    output[b * out_channels * output_height * output_width +
           oc * output_height * output_width +
           oy * output_width + ox] = acc;
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups) {

    auto input_sizes = input.sizes();
    int batch_size = input_sizes[0];
    int in_channels = input_sizes[1];
    int input_height = input_sizes[2];
    int input_width = input_sizes[3];

    auto weight_sizes = weight.sizes();
    int out_channels = weight_sizes[0];

    int output_height = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int output_width = output_height;

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, input.options());

    dim3 threads(8, 8, 1);
    dim3 blocks((output_width + 7) / 8, (output_height + 7) / 8, batch_size * out_channels);

    conv_transpose2d_kernel<<<blocks, threads>>>(
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
        output_padding,
        groups);

    return output;
}
"""

conv_transpose2d_cpp_source = """
torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups);
"""

# Compile the inline CUDA code for transposed convolution
conv_transpose2d = load_inline(
    name="conv_transpose2d",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_cuda_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom CUDA kernel for transposed 2D convolution.
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

        # Create weight tensor with standard layout: (out_channels, in_channels, kernel_size, kernel_size)
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        )

        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, mode="fan_in", nonlinearity="relu")

        self.cuda_conv_transpose2d = conv_transpose2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution using a custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return self.cuda_conv_transpose2d.conv_transpose2d_cuda(
            x.contiguous(),
            self.weight.contiguous(),
            self.kernel_size,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
        )
