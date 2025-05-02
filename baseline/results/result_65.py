import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for transposed convolution with asymmetric kernel (3x5)
conv_transpose_cuda_code = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Naive implementation of transposed convolution for asymmetric kernel (3,5)
__global__ void conv_transpose2d_kernel(
    const float* input,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int output_padding_h, int output_padding_w,
    const float* weight,
    const float* bias) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * out_h * out_w) return;

    int n = idx / (out_channels * out_h * out_w);
    int rest = idx % (out_channels * out_h * out_w);
    int c_out = rest / (out_h * out_w);
    int rest2 = rest % (out_h * out_w);
    int h_out = rest2 / out_w;
    int w_out = rest2 % out_w;

    float sum = 0.0f;

    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h_in = h_out * stride_h - padding_h + kh;
                int w_in = w_out * stride_w - padding_w + kw;

                if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
                    float input_val = input[n * in_channels * in_h * in_w + c_in * in_h * in_w + h_in * in_w + w_in];
                    float weight_idx = weight[c_out * in_channels * kernel_h * kernel_w + c_in * kernel_h * kernel_w + kh * kernel_w + kw];
                    sum += input_val * weight_idx;
                }
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[c_out];
    }

    output[n * out_channels * out_h * out_w + c_out * out_h * out_w + h_out * out_w + w_out] = sum;
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int output_padding_h, int output_padding_w) {

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);

    int out_channels = weight.size(0);
    int out_h = ((in_h - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h);
    int out_w = ((in_w - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w);

    auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, input.options());

    const int num_threads = 1024;
    const int num_blocks = (batch_size * out_channels * out_h * out_w + num_threads - 1) / num_threads;

    conv_transpose2d_kernel<<<num_blocks, num_threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_h, in_w,
        out_h, out_w,
        kernel_h, kernel_w,
        stride_h, stride_w,
        padding_h, padding_w,
        output_padding_h, output_padding_w,
        weight.data_ptr<float>(),
        bias.data_ptr<float>()
    );

    return output;
}
"""

conv_transpose_cpp_code = """
torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int output_padding_h, int output_padding_w);
"""

# Compile the inline CUDA code
conv_transpose_op = load_inline(
    name="conv_transpose2d",
    cpp_sources=conv_transpose_cpp_code,
    cuda_sources=conv_transpose_cuda_code,
    functions=["conv_transpose2d_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )
        self.kernel_h, self.kernel_w = kernel_size
        self.stride_h, self.stride_w = (
            stride,
            stride if isinstance(stride, int) else stride[0],
            stride[1] if isinstance(stride, tuple) else stride,
        )
        self.padding_h, self.padding_w = (
            padding,
            padding if isinstance(padding, int) else padding[0],
            padding[1] if isinstance(padding, tuple) else padding,
        )
        self.output_padding_h, self.output_padding_w = (
            output_padding,
            output_padding if isinstance(output_padding, int) else output_padding[0],
            output_padding[1] if isinstance(output_padding, tuple) else output_padding,
        )
        self.conv_transpose2d_cuda = conv_transpose_op.conv_transpose2d_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv_transpose2d.weight
        bias = (
            self.conv_transpose2d.bias
            if self.conv_transpose2d.bias is not None
            else torch.zeros(weight.size(0), device=x.device)
        )
        return self.conv_transpose2d_cuda(
            x,
            weight,
            bias,
            self.kernel_h,
            self.kernel_w,
            self.stride_h,
            self.stride_w,
            self.padding_h,
            self.padding_w,
            self.output_padding_h,
            self.output_padding_w,
        )


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a transposed 2D convolution with a square input and an asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (height, width).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Padding applied to the input. Defaults to 0.
        output_padding (int or tuple, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return self.conv_transpose2d(x)

# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = (3, 5)  # Asymmetric kernel
width = 128
height = 128

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization


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
