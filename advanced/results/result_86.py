import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA kernel for depthwise convolution
depthwise_conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void depthwise_conv2d_kernel(
    const float* input, 
    const float* weight, 
    float* output,
    int batch_size, 
    int in_channels, 
    int height, 
    int width, 
    int kernel_size, 
    int stride, 
    int padding, 
    int dilation) {
    
    int n = blockIdx.z;
    int c = blockIdx.y;
    int out_h = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_w = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    int h_start = threadIdx.x / out_w;
    int w_start = threadIdx.x % out_w;

    int out_idx = n * in_channels * out_h * out_w + c * out_h * out_w + h_start * out_w + w_start;
    float acc = 0.0f;

    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int h = h_start * stride - padding + dilation * kh;
            int w = w_start * stride - padding + dilation * kw;
            if (h >= 0 && h < height && w >= 0 && w < width) {
                int in_idx = n * in_channels * height * width + c * height * width + h * width + w;
                int w_idx = c * kernel_size * kernel_size + kh * kernel_size + kw;
                acc += input[in_idx] * weight[w_idx];
            }
        }
    }

    output[out_idx] = acc;
}

torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    int kernel_size, 
    int stride, 
    int padding, 
    int dilation) {

    auto size = input.sizes();
    int batch_size = size[0];
    int in_channels = size[1];
    int height = size[2];
    int width = size[3];

    int out_h = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_w = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    torch::Tensor output = torch::zeros({batch_size, in_channels, out_h, out_w}, options);

    dim3 blocks(1, in_channels, batch_size);
    dim3 threads(out_h * out_w);

    depthwise_conv2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        output.data_ptr<float>(),
        batch_size, 
        in_channels, 
        height, 
        width, 
        kernel_size, 
        stride, 
        padding, 
        dilation);

    return output;
}
"""

depthwise_conv2d_cpp_source = """
torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    int kernel_size, 
    int stride, 
    int padding, 
    int dilation);
"""

# Compile the inline CUDA code for depthwise convolution
depthwise_conv2d = load_inline(
    name="depthwise_conv2d",
    cpp_sources=depthwise_conv2d_cpp_source,
    cuda_sources=depthwise_conv2d_source,
    functions=["depthwise_conv2d_cuda"],
    verbose=False,
)

# CUDA kernel for pointwise convolution
pointwise_conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void pointwise_conv2d_kernel(
    const float* input, 
    const float* weight, 
    float* output,
    int batch_size, 
    int in_channels, 
    int out_channels, 
    int height, 
    int width) {
    
    int n = blockIdx.z;
    int h = blockIdx.y;
    int w = blockIdx.x;
    int c_out = threadIdx.x;

    float acc = 0.0f;

    for (int c_in = 0; c_in < in_channels; ++c_in) {
        int in_idx = n * in_channels * height * width + c_in * height * width + h * width + w;
        int w_idx = c_out * in_channels + c_in;
        acc += input[in_idx] * weight[w_idx];
    }

    output[n * out_channels * height * width + c_out * height * width + h * width + w] = acc;
}

torch::Tensor pointwise_conv2d_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    int out_channels) {

    auto size = input.sizes();
    int batch_size = size[0];
    int in_channels = size[1];
    int height = size[2];
    int width = size[3];

    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    torch::Tensor output = torch::zeros({batch_size, out_channels, height, width}, options);

    dim3 blocks(width, height, batch_size);
    dim3 threads(out_channels);

    pointwise_conv2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        output.data_ptr<float>(),
        batch_size, 
        in_channels, 
        out_channels, 
        height, 
        width);

    return output;
}
"""

pointwise_conv2d_cpp_source = """
torch::Tensor pointwise_conv2d_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    int out_channels);
"""

# Compile the inline CUDA code for pointwise convolution
pointwise_conv2d = load_inline(
    name="pointwise_conv2d",
    cpp_sources=pointwise_conv2d_cpp_source,
    cuda_sources=pointwise_conv2d_source,
    functions=["pointwise_conv2d_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom CUDA kernels for depthwise-separable 2D convolution.
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
        self.bias = bias

        # Depthwise weights and Pointwise weights
        self.depthwise_weight = nn.Parameter(
            torch.Tensor(in_channels, 1, kernel_size, kernel_size)
        )
        self.pointwise_weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, 1, 1)
        )

        # Custom CUDA ops
        self.depthwise_conv2d = depthwise_conv2d
        self.pointwise_conv2d = pointwise_conv2d

        # Initialize weights
        nn.init.kaiming_normal_(self.depthwise_weight)
        nn.init.kaiming_normal_(self.pointwise_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise-separable 2D convolution using custom CUDA kernels.
        """
        # Depthwise convolution
        x = self.depthwise_conv2d.depthwise_conv2d_cuda(
            x,
            self.depthwise_weight,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        )

        # Pointwise convolution
        x = self.pointwise_conv2d.pointwise_conv2d_cuda(
            x, self.pointwise_weight, self.out_channels
        )

        return x


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a depthwise-separable 2D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise-separable 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
width = 256
height = 256
stride = 1
padding = 0
dilation = 1

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]


# correctness
assert len(get_init_inputs()) == 0
inputs = get_inputs()
inputs = [x.cuda() for x in inputs]
model_result = Model()(*inputs)
model_new_result = ModelNew()(*inputs)
assert torch.allclose(model_result.detach().cpu(), model_new_result.detach().cpu(), rtol=1e-02, atol=1e-03)

# profiling
import triton.profiler as proton
from triton.testing import do_bench

def bench(func, warmup=0, repeat=10, proton_name="kernel"):
    with proton.scope(proton_name, metrics={}):
        ms = do_bench(func, warmup=warmup, rep=repeat)
    return ms

func_model = lambda: Model()(*inputs)
func_model_new = lambda: ModelNew()(*inputs)
model_ms = bench(func_model, warmup=0, repeat=10, proton_name="Model")
model_new_ms = bench(func_model_new, warmup=0, repeat=10, proton_name="ModelNew")
print(f"Model: {model_ms} ms")
print(f"ModelNew: {model_new_ms} ms")
