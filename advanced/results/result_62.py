import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 2D convolution with asymmetric kernel
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
    int kernel_h, 
    int kernel_w,
    int stride_h, 
    int stride_w,
    int padding_h, 
    int padding_w,
    int dilation_h, 
    int dilation_w
) {
    int oc = blockIdx.z;
    int b = blockIdx.y;
    int threadIdx_y = threadIdx.y;
    int threadIdx_x = threadIdx.x;

    int output_height = (input_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int output_width = (input_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    __shared__ float tile[32][34]; // For shared memory optimization

    float acc = 0.0f;

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h_offset = threadIdx_y * stride_h + kh * dilation_h - padding_h;
                int w_offset = threadIdx_x * stride_w + kw * dilation_w - padding_w;

                if (h_offset >= 0 && h_offset < input_height && w_offset >= 0 && w_offset < input_width) {
                    tile[threadIdx_y][threadIdx_x] = input[((b * in_channels + ic) * input_height + h_offset) * input_width + w_offset];
                } else {
                    tile[threadIdx_y][threadIdx_x] = 0.0f;
                }
                __syncthreads();

                float w = weight[((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw];
                acc += tile[threadIdx_y][threadIdx_x] * w;
                __syncthreads();
            }
        }
    }

    int oy = threadIdx_y;
    int ox = threadIdx_x;

    if (oy < output_height && ox < output_width) {
        output[(b * out_channels + oc) * output_height * output_width + oy * output_width + ox] = acc;
    }
}

torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight, int64_t kernel_h, int64_t kernel_w, int64_t stride_h, int64_t stride_w, int64_t padding_h, int64_t padding_w, int64_t dilation_h, int64_t dilation_w) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_height = input.size(2);
    auto input_width = input.size(3);
    auto out_channels = weight.size(0);

    int output_height = (input_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int output_width = (input_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, input.options());

    dim3 threads(16, 16);
    dim3 blocks((output_width + 15) / 16, batch_size, out_channels);

    conv2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels, input_height, input_width,
        kernel_h, kernel_w,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w
    );

    return output;
}
"""

conv2d_cpp_source = """
torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight, int64_t kernel_h, int64_t kernel_w, int64_t stride_h, int64_t stride_w, int64_t padding_h, int64_t padding_w, int64_t dilation_h, int64_t dilation_w);
"""

# Compile the inline CUDA code for convolution
conv2d_op = load_inline(
    name="conv2d_op",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_cuda_source,
    functions=["conv2d_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(
            torch.randn(
                out_channels, in_channels // groups, kernel_size[0], kernel_size[1]
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter("bias", None)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.conv2d_op = conv2d_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2d_op.conv2d_cuda(
            x,
            self.weight,
            self.kernel_size[0],
            self.kernel_size[1],
            self.stride,
            self.stride,
            self.padding,
            self.padding,
            self.dilation,
            self.dilation,
        )
