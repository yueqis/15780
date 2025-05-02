import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D transposed convolution
conv_transpose3d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose3d_kernel(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int depth,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int bias_exists,
    const float* bias) {

    // Simplified version of transposed convolution
    int oc = blockIdx.x;
    int b = blockIdx.y;

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    for (int od = 0; od < depth; ++od) {
                        for (int oh = 0; oh < height; ++oh) {
                            for (int ow = 0; ow < width; ++ow) {
                                int id = od * stride - padding + kd * dilation;
                                int ih = oh * stride - padding + kh * dilation;
                                int iw = ow * stride - padding + kw * dilation;

                                if (id >= 0 && id < depth && ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                    int input_idx = b * in_channels * depth * height * width + ic * depth * height * width + id * height * width + ih * width + iw;
                                    int weight_idx = oc * in_channels * kernel_size * kernel_size * kernel_size + ic * kernel_size * kernel_size * kernel_size + kd * kernel_size * kernel_size + kh * kernel_size + kw;
                                    int output_idx = b * out_channels * depth * height * width + oc * depth * height * width + od * height * width + oh * width + ow;

                                    atomicAdd(&output[output_idx], input[input_idx] * weight[weight_idx]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (bias_exists && bias != nullptr) {
        for (int od = 0; od < depth; ++od) {
            for (int oh = 0; oh < height; ++oh) {
                for (int ow = 0; ow < width; ++ow) {
                    int output_idx = b * out_channels * depth * height * width + oc * depth * height * width + od * height * width + oh * width + ow;
                    output[output_idx] += bias[oc];
                }
            }
        }
    }
}

torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    bool bias_exists) {

    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);

    auto output = torch::zeros_like(input, torch::MemoryFormat::Contiguous);
    
    dim3 blocks(out_channels, batch_size);
    dim3 threads(1);

    conv_transpose3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        depth,
        height,
        width,
        kernel_size,
        stride,
        padding,
        dilation,
        bias_exists,
        bias_exists ? bias.data_ptr<float>() : nullptr);

    return output;
}
"""

conv_transpose3d_cpp_source = """
torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    bool bias_exists);
"""

# Compile the inline CUDA code
conv_transpose3d_op = load_inline(
    name="conv_transpose3d_op",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_cuda_source,
    functions=["conv_transpose3d_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized version of Model using a custom CUDA kernel for ConvTranspose3d.
    """

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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias_enabled = bias

        # Register weights and bias as parameters
        self.weight = nn.Parameter(
            torch.randn(
                out_channels, in_channels, kernel_size, kernel_size, kernel_size
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter("bias", None)

        # Register the custom CUDA operator
        self.conv_transpose3d_cuda = conv_transpose3d_op.conv_transpose3d_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_transpose3d_cuda(
            x,
            self.weight,
            self.bias if self.bias_enabled else torch.tensor([]),
            self.stride,
            self.padding,
            self.dilation,
            self.bias_enabled,
        )
