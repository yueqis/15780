import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for MaxPool3d
maxpool3d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void maxpool3d_kernel(
    const scalar_t* input,
    scalar_t* output,
    int batch_size, int channels, int in_depth, int in_height, int in_width,
    int out_depth, int out_height, int out_width,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w) {

    int n = blockIdx.z;
    int c = blockIdx.y;
    int od = blockIdx.x / (out_height * out_width);
    int oh = (blockIdx.x / out_width) % out_height;
    int ow = blockIdx.x % out_width;

    int id_start = od * stride_d - pad_d;
    int ih_start = oh * stride_h - pad_h;
    int iw_start = ow * stride_w - pad_w;

    scalar_t max_val = -1.0f / 0.0f; // Negative infinity

    for (int kd = 0; kd < kernel_d; ++kd) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int id = id_start + kd * dilation_d;
                int ih = ih_start + kh * dilation_h;
                int iw = iw_start + kw * dilation_w;

                if (id >= 0 && id < in_depth && ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                    scalar_t val = input[(n * channels + c) * in_depth * in_height * in_width + id * in_height * in_width + ih * in_width + iw];
                    if (val > max_val) {
                        max_val = val;
                    }
                }
            }
        }
    }

    output[(n * channels + c) * out_depth * out_height * out_width + od * out_height * out_width + oh * out_width + ow] = max_val;
}

torch::Tensor maxpool3d_cuda(torch::Tensor input,
                             int kernel_d, int kernel_h, int kernel_w,
                             int stride_d, int stride_h, int stride_w,
                             int pad_d, int pad_h, int pad_w,
                             int dilation_d, int dilation_h, int dilation_w) {
    auto input_sizes = input.sizes();
    int batch_size = input_sizes[0];
    int channels = input_sizes[1];
    int in_depth = input_sizes[2];
    int in_height = input_sizes[3];
    int in_width = input_sizes[4];

    int out_depth = (in_depth + 2 * pad_d - dilation_d * (kernel_d - 1) - 1) / stride_d + 1;
    int out_height = (in_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_width = (in_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    auto output = torch::zeros({batch_size, channels, out_depth, out_height, out_width}, input.options());

    dim3 grid_dim(out_depth * out_height * out_width); // Each block handles one output element
    dim3 block_dim(1); // No threading inside block here, can be optimized further

    AT_DISPATCH_FLOATING_TYPES(input.type(), "maxpool3d_cuda", ([&] {
        maxpool3d_kernel<scalar_t><<<grid_dim, block_dim>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, channels, in_depth, in_height, in_width,
            out_depth, out_height, out_width,
            kernel_d, kernel_h, kernel_w,
            stride_d, stride_h, stride_w,
            pad_d, pad_h, pad_w,
            dilation_d, dilation_h, dilation_w);
    }));

    return output;
}
"""

maxpool3d_cpp_source = """
torch::Tensor maxpool3d_cuda(torch::Tensor input,
                             int kernel_d, int kernel_h, int kernel_w,
                             int stride_d, int stride_h, int stride_w,
                             int pad_d, int pad_h, int pad_w,
                             int dilation_d, int dilation_h, int dilation_w);
"""

# Compile the inline CUDA code
custom_maxpool3d = load_inline(
    name="custom_maxpool3d",
    cpp_sources=maxpool3d_cpp_source,
    cuda_sources=maxpool3d_cuda_source,
    functions=["maxpool3d_cuda"],
    verbose=False,
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
        self.maxpool3d_cuda = custom_maxpool3d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            x = x.cuda()
        return self.maxpool3d_cuda.maxpool3d_cuda(
            x,
            self.kernel_size,
            self.kernel_size,
            self.kernel_size,
            self.stride,
            self.stride,
            self.stride,
            self.padding,
            self.padding,
            self.padding,
            self.dilation,
            self.dilation,
            self.dilation,
        )


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Max Pooling 3D.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        """
        Initializes the Max Pooling 3D layer.

        Args:
            kernel_size (int): Size of the kernel for the max pooling operation.
            stride (int, optional): Stride of the pooling operation. Defaults to None, which means stride is equal to kernel_size.
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
            return_indices (bool, optional): Whether to return indices of the maximum values. Defaults to False.
            ceil_mode (bool, optional): When True, the output size is ceil(input_size / stride) instead of floor. Defaults to False.
        """
        super(Model, self).__init__()
        self.maxpool = nn.MaxPool3d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 3D to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, dim1, dim2, dim3).

        Returns:
            torch.Tensor: Output tensor with Max Pooling 3D applied.
        """
        return self.maxpool(x)

batch_size = 16
channels = 32
dim1 = 64
dim2 = 64
dim3 = 64
kernel_size = 3
stride = 2
padding = 1
dilation = 3

def get_inputs():
    x = torch.randn(batch_size, channels, dim1, dim2, dim3)
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation]
