import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for MaxPool3D
maxpool3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define DIV_UP(a, b) (((a) + (b) - 1) / (b))

__global__ void maxpool3d_kernel(
    const float* input,
    float* output,
    int batch_size, int channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size, int stride, int padding, int dilation) {

    int n = blockIdx.z;
    int c = blockIdx.y;
    int od = threadIdx.z;
    int oh = threadIdx.y;
    int ow = threadIdx.x;

    int id_start = od * stride - padding + dilation * 0;
    int ih_start = oh * stride - padding + dilation * 0;
    int iw_start = ow * stride - padding + dilation * 0;

    float max_val = -1.0 / 0.0; // negative infinity

    for (int kd = 0; kd < kernel_size; ++kd) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int id = id_start + kd * dilation;
                int ih = ih_start + kh * dilation;
                int iw = iw_start + kw * dilation;

                if (id >= 0 && id < in_d && ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                    int in_idx = n * channels * in_d * in_h * in_w +
                                 c * in_d * in_h * in_w +
                                 id * in_h * in_w +
                                 ih * in_w +
                                 iw;
                    max_val = fmaxf(max_val, input[in_idx]);
                }
            }
        }
    }

    int out_idx = n * channels * out_d * out_h * out_w +
                  c * out_d * out_h * out_w +
                  od * out_h * out_w +
                  oh * out_w +
                  ow;
    output[out_idx] = max_val;
}

torch::Tensor maxpool3d_cuda(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {

    auto in_sizes = input.sizes();
    int batch_size = in_sizes[0];
    int channels = in_sizes[1];
    int in_d = in_sizes[2];
    int in_h = in_sizes[3];
    int in_w = in_sizes[4];

    int out_d = (in_d + stride - 1) / stride;
    int out_h = (in_h + stride - 1) / stride;
    int out_w = (in_w + stride - 1) / stride;

    auto options = input.options().dtype(torch::kFloat32);
    torch::Tensor output = torch::zeros({batch_size, channels, out_d, out_h, out_w}, options);

    dim3 block(8, 8, 8); // Threads per block
    dim3 grid(channels, batch_size, out_d); // Blocks per grid

    maxpool3d_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        kernel_size, stride, padding, dilation);

    return output;
}
"""

maxpool3d_cpp_source = """
torch::Tensor maxpool3d_cuda(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation);
"""

# Compile the inline CUDA code
maxpool3d_op = load_inline(
    name="maxpool3d",
    cpp_sources=maxpool3d_cpp_source,
    cuda_sources=maxpool3d_source,
    functions=["maxpool3d_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        stride: int = None,
        padding: int = 0,
        dilation: int = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        self.maxpool_cuda = maxpool3d_op.maxpool3d_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use the custom CUDA kernel
        return self.maxpool_cuda(
            x, self.kernel_size, self.stride, self.padding, self.dilation
        )
