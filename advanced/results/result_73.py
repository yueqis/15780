import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D transposed convolution
conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose3d_kernel(
    const float* input, 
    const float* weight, 
    const float* bias,
    float* output,
    int batch_size, int in_channels, int out_channels,
    int depth, int height, int width,
    int kernel_size, int stride, int padding,
    int groups) {

    // Simple implementation for demonstration; real-world use requires optimization
    int n = blockIdx.z;
    int oc = blockIdx.y;
    int g = oc / (out_channels / groups);

    int d_start = threadIdx.x;
    int h_start = threadIdx.y;
    int w_start = threadIdx.z;

    int oc_group = oc % (out_channels / groups);
    int ic_group = oc % (in_channels / groups);

    float acc = 0.0f;

    for (int kd = 0; kd < kernel_size; ++kd) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int d_in = d_start * stride - padding + kd;
                int h_in = h_start * stride - padding + kh;
                int w_in = w_start * stride - padding + kw;

                if (d_in >= 0 && d_in < depth && h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    int input_idx = n * in_channels * depth * height * width +
                                    (g * (in_channels / groups) + ic_group) * depth * height * width +
                                    d_in * height * width +
                                    h_in * width +
                                    w_in;

                    int weight_idx = oc * in_channels * kernel_size * kernel_size * kernel_size +
                                     (g * (in_channels / groups) + ic_group) * kernel_size * kernel_size * kernel_size +
                                     kd * kernel_size * kernel_size +
                                     kh * kernel_size +
                                     kw;

                    acc += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    int output_idx = n * out_channels * depth * height * width +
                     oc * depth * height * width +
                     d_start * height * width +
                     h_start * width +
                     w_start;

    output[output_idx] = bias ? acc + bias[oc] : acc;
}

torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias,
    int batch_size, int in_channels, int out_channels,
    int depth, int height, int width,
    int kernel_size, int stride, int padding,
    int groups) {

    auto output = torch::zeros_like(input);

    dim3 threads(8, 8, 8);
    dim3 blocks((depth + threads.x - 1) / threads.x,
                (height + threads.y - 1) / threads.y,
                (width + threads.z - 1) / threads.z);

    conv_transpose3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        depth, height, width,
        kernel_size, stride, padding,
        groups);

    return output;
}
"""

conv_transpose3d_cpp_source = """
torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias,
    int batch_size, int in_channels, int out_channels,
    int depth, int height, int width,
    int kernel_size, int stride, int padding,
    int groups);
"""

# Compile the inline CUDA code for 3D transposed convolution
conv_transpose3d_op = load_inline(
    name="conv_transpose3d",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.weight = nn.Parameter(
            torch.randn(
                out_channels,
                in_channels // groups,
                kernel_size,
                kernel_size,
                kernel_size,
            )
        )
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None
        self.conv_transpose3d_op = conv_transpose3d_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, depth, height, width = x.shape
        return self.conv_transpose3d_op.conv_transpose3d_cuda(
            x,
            self.weight,
            self.bias,
            batch_size,
            self.in_channels,
            self.out_channels,
            depth,
            height,
            width,
            self.kernel_size,
            self.stride,
            self.padding,
            self.groups,
        )


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a 3D transposed convolution operation with asymmetric input and square kernel.
    The input is padded before the convolution.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size, kernel_size), stride=stride, padding=padding, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D transposed convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        return self.conv_transpose3d(x)

# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
depth = 16
height = 32
width = 32
stride = 2
padding = 3
groups = 4

def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, groups]


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
