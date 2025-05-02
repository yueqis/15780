import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for MaxPool2d
maxpool2d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void maxpool2d_kernel(
    const scalar_t* input, scalar_t* output,
    int batch_size, int channels, int height, int width,
    int kernel_h, int kernel_w, int stride_h, int stride_w,
    int padding_h, int padding_w, int dilation_h, int dilation_w,
    int out_height, int out_width) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;

    if (i >= out_width || j >= out_height || c >= channels)
        return;

    int batch = blockIdx.z / channels;
    int channel_in_batch = blockIdx.z % channels;

    scalar_t max_val = -1.0 / 0.0; // Negative infinity

    int input_offset = (batch * channels + channel_in_batch) * height * width;

    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            int h_start = j * stride_h - padding_h + kh * dilation_h;
            int w_start = i * stride_w - padding_w + kw * dilation_w;

            if (h_start >= 0 && w_start >= 0 && h_start < height && w_start < width) {
                scalar_t val = input[input_offset + h_start * width + w_start];
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
    }

    int output_idx = (batch * channels + channel_in_batch) * out_height * out_width + j * out_width + i;
    output[output_idx] = max_val;
}

torch::Tensor maxpool2d_cuda(torch::Tensor input,
                             int kernel_h, int kernel_w,
                             int stride_h, int stride_w,
                             int padding_h, int padding_w,
                             int dilation_h, int dilation_w) {
    auto input_sizes = input.sizes();
    int batch_size = input_sizes[0];
    int channels = input_sizes[1];
    int height = input_sizes[2];
    int width = input_sizes[3];

    int out_height = ((height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h) + 1;
    int out_width = ((width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w) + 1;

    auto output = torch::empty({batch_size, channels, out_height, out_width}, input.options());

    dim3 blocks(out_width, out_height, batch_size * channels);
    dim3 threads(8, 8);

    if (input.dtype() == torch::kFloat32) {
        maxpool2d_kernel<float><<<blocks, threads>>>(
            input.data_ptr<float>(), output.data_ptr<float>(),
            batch_size, channels, height, width,
            kernel_h, kernel_w, stride_h, stride_w,
            padding_h, padding_w, dilation_h, dilation_w,
            out_height, out_width);
    } else if (input.dtype() == torch::kFloat16) {
        maxpool2d_kernel<__half><<<blocks, threads>>>(
            reinterpret_cast<const __half*>(input.data_ptr<__half>()),
            reinterpret_cast<__half*>(output.data_ptr<__half>()),
            batch_size, channels, height, width,
            kernel_h, kernel_w, stride_h, stride_w,
            padding_h, padding_w, dilation_h, dilation_w,
            out_height, out_width);
    }

    return output;
}
"""

maxpool2d_cpp_source = """
torch::Tensor maxpool2d_cuda(torch::Tensor input,
                             int kernel_h, int kernel_w,
                             int stride_h, int stride_w,
                             int padding_h, int padding_w,
                             int dilation_h, int dilation_w);
"""

# Compile the inline CUDA code for custom MaxPool2d
custom_maxpool = load_inline(
    name="custom_maxpool",
    cpp_sources=maxpool2d_cpp_source,
    cuda_sources=maxpool2d_cuda_source,
    functions=["maxpool2d_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.maxpool_op = custom_maxpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Move tensor to GPU if not already there
        if not x.is_cuda:
            x = x.cuda()
        return self.maxpool_op.maxpool2d_cuda(
            x,
            self.kernel_size,
            self.kernel_size,
            self.stride,
            self.stride,
            self.padding,
            self.padding,
            self.dilation,
            self.dilation,
        )
