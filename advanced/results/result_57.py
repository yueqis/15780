import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for transposed 2D convolution
conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose2d_kernel(
    const float* input, float* output,
    int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size, int stride,
    int padding, int output_padding,
    const float* weight, const float* bias) {
    
    // Calculate global thread index
    int n = blockIdx.z; // Batch index
    int oh = blockIdx.y * blockDim.y + threadIdx.y; // Output height index
    int ow = blockIdx.x * blockDim.x + threadIdx.x; // Output width index
    int och = blockIdx.w; // Output channel index

    if (n >= batch_size || oh >= (height - 1) * stride + kernel_size - 2 * padding + output_padding ||
        ow >= (width - 1) * stride + kernel_size - 2 * padding + output_padding || och >= out_channels) {
        return;
    }

    // Initialize output to zero or bias
    output[n * out_channels * ((height - 1) * stride + kernel_size - 2 * padding + output_padding) * ((width - 1) * stride + kernel_size - 2 * padding + output_padding) +
           och * ((height - 1) * stride + kernel_size - 2 * padding + output_padding) * ((width - 1) * stride + kernel_size - 2 * padding + output_padding) +
           oh * ((width - 1) * stride + kernel_size - 2 * padding + output_padding) + ow] =
        (bias != nullptr) ? bias[och] : 0.0f;

    // Perform transposed convolution
    for (int kch = 0; kch < in_channels; ++kch) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int ih = oh * stride - padding + kh;
                int iw = ow * stride - padding + kw;

                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    float val = input[n * in_channels * height * width + kch * height * width + ih * width + iw];
                    float wgt = weight[och * in_channels * kernel_size * kernel_size + kch * kernel_size * kernel_size + kh * kernel_size + kw];
                    output[n * out_channels * ((height - 1) * stride + kernel_size - 2 * padding + output_padding) * ((width - 1) * stride + kernel_size - 2 * padding + output_padding) +
                           och * ((height - 1) * stride + kernel_size - 2 * padding + output_padding) * ((width - 1) * stride + kernel_size - 2 * padding + output_padding) +
                           oh * ((width - 1) * stride + kernel_size - 2 * padding + output_padding) + ow] += val * wgt;
                }
            }
        }
    }
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int64_t stride, int64_t padding, int64_t output_padding) {
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);

    int out_height = (height - 1) * stride + kernel_size - 2 * padding + output_padding;
    int out_width = (width - 1) * stride + kernel_size - 2 * padding + output_padding;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

    dim3 threads(16, 16); // Threads per block
    dim3 blocks((out_width + 15) / 16, (out_height + 15) / 16, batch_size, out_channels);

    conv_transpose2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, in_channels, out_channels, height, width, kernel_size,
        stride, padding, output_padding,
        weight.data_ptr<float>(), bias.has_value() ? bias->data_ptr<float>() : nullptr);

    return output;
}
"""

conv_transpose2d_cpp_source = (
    "torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, "
    "int64_t stride, int64_t padding, int64_t output_padding);"
)

# Compile the inline CUDA code
conv_transpose2d_op = load_inline(
    name="conv_transpose2d",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Custom implementation of ConvTranspose2d using a custom CUDA kernel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
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
            torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.Tensor(out_channels)) if bias else None

        # Initialize weights and biases
        nn.init.kaiming_uniform_(self.weight, nonlinearity="leaky_relu", a=0.2)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv_transpose2d_op.conv_transpose2d_cuda(
            x, self.weight, self.bias, self.stride, self.padding, self.output_padding
        )


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a transposed 2D convolution with square input and square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        output_padding (int, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
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
kernel_size = 3
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
