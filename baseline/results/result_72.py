import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for 3D transposed convolution
conv_transpose3d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Simplified 3D transposed convolution kernel (basic implementation)
__global__ void conv_transpose3d_kernel(
    const float* input, float* output,
    const float* weight, const float* bias,
    int batch_size, int in_channels, int out_channels,
    int depth, int height, int width,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int groups) {

    // Compute output dimensions manually based on parameters
    int out_depth = (depth - 1) * stride_d - 2 * padding_d + kernel_d + output_padding_d;
    int out_height = (height - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h;
    int out_width = (width - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w;

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= batch_size * out_channels * out_depth * out_height * out_width) return;

    int w_offset = gid % out_width; gid /= out_width;
    int h_offset = gid % out_height; gid /= out_height;
    int d_offset = gid % out_depth; gid /= out_depth;
    int oc = gid % out_channels; gid /= out_channels;
    int b = gid;

    int group_id = oc / (out_channels / groups);
    int group_in_channels = in_channels / groups;
    int group_weight_offset = group_id * (out_channels / groups) * group_in_channels * kernel_d * kernel_h * kernel_w;

    float acc = 0.0f;
    if (bias != nullptr) {
        acc = bias[oc];
    }

    int ic_start = group_id * group_in_channels;
    int ic_end = ic_start + group_in_channels;

    for (int ic = ic_start; ic < ic_end; ++ic) {
        for (int kd = 0; kd < kernel_d; ++kd) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int in_d = d_offset / stride_d - padding_d + kd;
                    int in_h = h_offset / stride_h - padding_h + kh;
                    int in_w = w_offset / stride_w - padding_w + kw;

                    if (d_offset % stride_d != 0 || h_offset % stride_h != 0 || w_offset % stride_w != 0) {
                        continue;
                    }

                    if (in_d < 0 || in_d >= depth || in_h < 0 || in_h >= height || in_w < 0 || in_w >= width) {
                        continue;
                    }

                    int in_idx = b * in_channels * depth * height * width +
                                 ic * depth * height * width +
                                 in_d * height * width +
                                 in_h * width +
                                 in_w;

                    int weight_idx = group_weight_offset +
                                     oc * group_in_channels * kernel_d * kernel_h * kernel_w +
                                     (ic - ic_start) * kernel_d * kernel_h * kernel_w +
                                     kd * kernel_h * kernel_w +
                                     kh * kernel_w +
                                     kw;

                    acc += input[in_idx] * weight[weight_idx];
                }
            }
        }
    }

    int out_idx = b * out_channels * out_depth * out_height * out_width +
                  oc * out_depth * out_height * out_width +
                  d_offset * out_height * out_width +
                  h_offset * out_width +
                  w_offset;

    output[out_idx] = acc;
}

torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int64_t stride_d, int64_t stride_h, int64_t stride_w,
    int64_t padding_d, int64_t padding_h, int64_t padding_w,
    int64_t output_padding_d, int64_t output_padding_h, int64_t output_padding_w,
    int64_t groups) {

    auto input_sizes = input.sizes();
    int batch_size = input_sizes[0];
    int in_channels = input_sizes[1];
    int depth = input_sizes[2];
    int height = input_sizes[3];
    int width = input_sizes[4];

    int out_channels = weight.size(0);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);

    int out_depth = (depth - 1) * stride_d - 2 * padding_d + kernel_d + output_padding_d;
    int out_height = (height - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h;
    int out_width = (width - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w;

    auto output = torch::zeros({batch_size, out_channels, out_depth, out_height, out_width}, input.options());

    int total_threads = batch_size * out_channels * out_depth * out_height * out_width;
    int block_size = 256;
    int num_blocks = (total_threads + block_size - 1) / block_size;

    conv_transpose3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        weight.data_ptr<float>(), bias ? bias.data_ptr<float>() : nullptr,
        batch_size, in_channels, out_channels,
        depth, height, width,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w,
        groups);

    return output;
}
"""

conv_transpose3d_cpp_source = """
torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int64_t stride_d, int64_t stride_h, int64_t stride_w,
    int64_t padding_d, int64_t padding_h, int64_t padding_w,
    int64_t output_padding_d, int64_t output_padding_h, int64_t output_padding_w,
    int64_t groups);
"""

# Compile the inline CUDA code
conv_transpose3d_op = load_inline(
    name="conv_transpose3d",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_cuda_source,
    functions=["conv_transpose3d_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1, 1),
        padding: tuple = (0, 0, 0),
        output_padding: tuple = (0, 0, 0),
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
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        # Initialize weights and biases
        nn.init.kaiming_uniform_(self.weight, mode="fan_out", nonlinearity="relu")
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        self.cuda_op = conv_transpose3d_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cuda_op.conv_transpose3d_cuda(
            x,
            self.weight,
            self.bias,
            self.stride[0],
            self.stride[1],
            self.stride[2],
            self.padding[0],
            self.padding[1],
            self.padding[2],
            self.output_padding[0],
            self.output_padding[1],
            self.output_padding[2],
            self.groups,
        )
