import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for ConvTranspose2d with asymmetric padding and kernel size
conv_transpose2d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose2d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kh,
    int kw,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w) {

    int b = blockIdx.z;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;

    if (oh >= out_height || ow >= out_width) return;

    int oy = oh * stride_h;
    int ox = ow * stride_w;

    float sum = 0.0f;
    for (int c = 0; c < in_channels; ++c) {
        for (int ky = 0; ky < kh; ++ky) {
            for (int kx = 0; kx < kw; ++kx) {
                int iy = oy - pad_h + ky;
                int ix = ox - pad_w + kx;
                if (iy >= 0 && iy < in_height && ix >= 0 && ix < in_width) {
                    float w_val = weight[(c * out_channels * kh * kw) + (kx + kw * (ky + kh * (b % out_channels)))];
                    float i_val = input[b * in_channels * in_height * in_width + c * in_height * in_width + iy * in_width + ix];
                    sum += w_val * i_val;
                }
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[b % out_channels];
    }

    output[b * out_channels * out_height * out_width + (b % out_channels) * out_height * out_width + oh * out_width + ow] = sum;
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kh, int kw,
    int stride_h, int stride_w,
    int pad_h, int pad_w) {

    auto input_sizes = input.sizes();
    int batch_size = input_sizes[0];
    int in_channels = input_sizes[1];
    int in_height = input_sizes[2];
    int in_width = input_sizes[3];

    auto weight_sizes = weight.sizes();
    int out_channels = weight_sizes[0];

    int out_height = in_height * stride_h + ((kh - 1) - 2 * pad_h);
    int out_width = in_width * stride_w + ((kw - 1) - 2 * pad_w);

    auto options = input.options();
    torch::Tensor output = torch::zeros({batch_size * out_channels, out_channels, out_height, out_width}, options);

    dim3 block(16, 16);
    dim3 grid((out_width + block.x - 1) / block.x, (out_height + block.y - 1) / block.y, batch_size * out_channels);

    conv_transpose2d_kernel<<<grid, block>>>(input.data_ptr<float>(),
                                            weight.data_ptr<float>(),
                                            bias.defined() ? bias.data_ptr<float>() : nullptr,
                                            output.data_ptr<float>(),
                                            batch_size,
                                            in_channels,
                                            out_channels,
                                            in_height,
                                            in_width,
                                            out_height,
                                            out_width,
                                            kh,
                                            kw,
                                            stride_h,
                                            stride_w,
                                            pad_h,
                                            pad_w);

    return output;
}
"""

conv_transpose2d_cpp_source = """
torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kh, int kw,
    int stride_h, int stride_w,
    int pad_h, int pad_w);
"""

# Compile the inline CUDA code
conv_transpose2d_op = load_inline(
    name="conv_transpose2d",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_cuda_source,
    functions=["conv_transpose2d_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(
            torch.Tensor(in_channels, out_channels, kernel_size[0], kernel_size[1])
        )
        self.bias = nn.Parameter(torch.Tensor(out_channels)) if bias else None
        self.reset_parameters()

        # Load custom CUDA op
        self.conv_transpose2d_cuda = conv_transpose2d_op.conv_transpose2d_cuda

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=0)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_transpose2d_cuda(
            x,
            self.weight,
            self.bias if self.bias is not None else torch.tensor([]),
            self.kernel_size[0],
            self.kernel_size[1],
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1],
        )
