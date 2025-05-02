import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused depthwise and pointwise convolution
depthwise_pointwise_conv2d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void depthwise_pointwise_conv2d_kernel(
    const float* input, 
    const float* depthwise_weight, 
    const float* pointwise_weight,
    const float* bias,
    float* output,
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width,
    int output_height, int output_width,
    int kernel_size, int stride, int padding, int dilation) {
    
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z * blockDim.z + threadIdx.z;

    if (ox >= output_width || oy >= output_height || b >= batch_size)
        return;

    int oy_start = oy * stride - padding;
    int ox_start = ox * stride - padding;

    // Temporary storage for depthwise output
    float* dw_out = (float*)malloc(in_channels * sizeof(float));
    if (!dw_out) return; // Handle allocation failure if necessary

    for (int ic = 0; ic < in_channels; ++ic) {
        float val = 0.0f;
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int iy = oy_start + ky * dilation;
                int ix = ox_start + kx * dilation;

                if (iy >= 0 && iy < input_height && ix >= 0 && ix < input_width) {
                    int input_idx = b * in_channels * input_height * input_width +
                                    ic * input_height * input_width +
                                    iy * input_width + ix;
                    
                    int weight_idx = ic * kernel_size * kernel_size + ky * kernel_size + kx;
                    val += input[input_idx] * depthwise_weight[weight_idx];
                }
            }
        }
        dw_out[ic] = val;  // No ReLU or other activation assumed
    }

    // Pointwise convolution
    for (int oc = 0; oc < out_channels; ++oc) {
        float val = 0.0f;
        for (int ic = 0; ic < in_channels; ++ic) {
            val += dw_out[ic] * pointwise_weight[oc * in_channels + ic];
        }
        if (bias) {
            val += bias[oc];
        }
        int output_idx = b * out_channels * output_height * output_width +
                         oc * output_height * output_width +
                         oy * output_width + ox;
        output[output_idx] = val;
    }

    free(dw_out);
}
"""

depthwise_pointwise_conv2d_cpp_source = """
torch::Tensor depthwise_pointwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor depthwise_weight,
    torch::Tensor pointwise_weight,
    torch::Tensor bias,
    int stride, int padding, int dilation);
"""

# Compile the inline CUDA code
conv2d_fused_op = load_inline(
    name="depthwise_pointwise_conv2d",
    cpp_sources=depthwise_pointwise_conv2d_cpp_source,
    cuda_sources=depthwise_pointwise_conv2d_cuda_source,
    functions=["depthwise_pointwise_conv2d_cuda"],
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
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.fused_conv = conv2d_fused_op.depthwise_pointwise_conv2d_cuda
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias_flag = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get weights and biases
        depthwise_weight = self.depthwise.weight
        pointwise_weight = self.pointwise.weight
        bias = self.pointwise.bias if self.bias_flag else torch.tensor([]).to(x.device)

        # Ensure inputs are contiguous
        x = x.contiguous()
        depthwise_weight = depthwise_weight.contiguous()
        pointwise_weight = pointwise_weight.contiguous()
        bias = bias.contiguous() if self.bias_flag else bias

        batch_size, _, input_height, input_width = x.shape
        in_channels = self.depthwise.in_channels
        out_channels = self.pointwise.out_channels
        kernel_size = self.depthwise.kernel_size[0]
        stride = self.depthwise.stride[0]
        padding = self.depthwise.padding[0]
        dilation = self.depthwise.dilation[0]

        # Compute output dimensions
        output_height = (
            input_height + 2 * padding - dilation * (kernel_size - 1) - 1
        ) // stride + 1
        output_width = (
            input_width + 2 * padding - dilation * (kernel_size - 1) - 1
        ) // stride + 1

        output = self.fused_conv(
            x, depthwise_weight, pointwise_weight, bias, stride, padding, dilation
        )

        # Reshape output to match expected shape
        return output.view(batch_size, out_channels, output_height, output_width)
