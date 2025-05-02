import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for transposed 3D convolution
conv_transpose3d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose3d_kernel(
    const float* input, float* output,
    const float* weight, const float* bias,
    int batch_size, int in_channels, int out_channels,
    int depth_in, int height_in, int width_in,
    int depth_out, int height_out, int width_out,
    int kD, int kH, int kW,
    int strideD, int strideH, int strideW,
    int padD, int padH, int padW,
    int output_padD, int output_padH, int output_padW,
    int groups) {

    // Compute thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * depth_out * height_out * width_out) return;

    int n = (idx / (out_channels * depth_out * height_out * width_out)) % batch_size;
    int c_out = (idx / (depth_out * height_out * width_out)) % out_channels;
    int d_out = (idx / (height_out * width_out)) % depth_out;
    int h_out = (idx / width_out) % height_out;
    int w_out = idx % width_out;

    int group = c_out / (out_channels / groups);
    int c_out_group = c_out % (out_channels / groups);
    int c_in_group = c_out_group * (in_channels / groups);

    float sum = 0.0f;

    for (int c_in = c_in_group; c_in < c_in_group + (in_channels / groups); ++c_in) {
        for (int kd = 0; kd < kD; ++kd) {
            for (int kh = 0; kh < kH; ++kh) {
                for (int kw = 0; kw < kW; ++kw) {
                    int d_in = d_out * strideD - padD + kd;
                    int h_in = h_out * strideH - padH + kh;
                    int w_in = w_out * strideW - padW + kw;

                    if (d_in >= 0 && d_in < depth_in &&
                        h_in >= 0 && h_in < height_in &&
                        w_in >= 0 && w_in < width_in) {
                        
                        int weight_idx = c_out * in_channels * kD * kH * kW +
                                         c_in * kD * kH * kW +
                                         kd * kH * kW + kh * kW + kw;

                        int input_idx = n * in_channels * depth_in * height_in * width_in +
                                        c_in * depth_in * height_in * width_in +
                                        d_in * height_in * width_in +
                                        h_in * width_in + w_in;

                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[c_out];
    }

    int output_idx = n * out_channels * depth_out * height_out * width_out +
                     c_out * depth_out * height_out * width_out +
                     d_out * height_out * width_out +
                     h_out * width_out + w_out;

    output[output_idx] = sum;
}

torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::IntArrayRef stride, torch::IntArrayRef padding, torch::IntArrayRef output_padding,
    int64_t groups) {
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int depth_in = input.size(2);
    int height_in = input.size(3);
    int width_in = input.size(4);
    int out_channels = weight.size(0);
    int kD = weight.size(2);
    int kH = weight.size(3);
    int kW = weight.size(4);

    int strideD = stride[0];
    int strideH = stride[1];
    int strideW = stride[2];

    int padD = padding[0];
    int padH = padding[1];
    int padW = padding[2];

    int output_padD = output_padding[0];
    int output_padH = output_padding[1];
    int output_padW = output_padding[2];

    int depth_out = (depth_in - 1) * strideD - 2 * padD + kD + output_padD;
    int height_out = (height_in - 1) * strideH - 2 * padH + kH + output_padH;
    int width_out = (width_in - 1) * strideW - 2 * padW + kW + output_padW;

    auto output = torch::zeros({batch_size, out_channels, depth_out, height_out, width_out}, input.options());

    const int threads_per_block = 256;
    const int num_blocks = (output.numel() + threads_per_block - 1) / threads_per_block;

    conv_transpose3d_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        weight.data_ptr<float>(), bias.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        depth_in, height_in, width_in,
        depth_out, height_out, width_out,
        kD, kH, kW,
        strideD, strideH, strideW,
        padD, padH, padW,
        output_padD, output_padH, output_padW,
        groups);

    return output;
}
"""

conv_transpose3d_cpp_source = """
torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::IntArrayRef stride, torch::IntArrayRef padding, torch::IntArrayRef output_padding,
    int64_t groups);
"""

# Compile the inline CUDA code
conv_transpose3d_op = load_inline(
    name="conv_transpose3d",
    cpp_sources=[conv_transpose3d_cpp_source],
    cuda_sources=[conv_transpose3d_cuda_source],
    functions=["conv_transpose3d_cuda"],
    verbose=True,
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

        # Create weight and bias parameters
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        # Initialize weights and biases
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        self.conv_transpose3d = conv_transpose3d_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_transpose3d.conv_transpose3d_cuda(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
        )


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a transposed 3D convolution operation with asymmetric input and kernel sizes.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Tuple of 3 integers representing the kernel size in the form (depth, height, width).
        stride (tuple, optional): Tuple of 3 integers representing the stride in the form (depth, height, width). Defaults to (1, 1, 1).
        padding (tuple, optional): Tuple of 3 integers representing the padding in the form (depth, height, width). Defaults to (0, 0, 0).
        output_padding (tuple, optional): Tuple of 3 integers representing the output padding in the form (depth, height, width). Defaults to (0, 0, 0).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth_in, height_in, width_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        return self.conv_transpose3d(x)

# Test code
batch_size = 16
in_channels = 32
out_channels = 16
kernel_size = (3, 5, 7)  # Asymmetric kernel size
depth_in = 16
height_in = 32
width_in = 64

def get_inputs():
    x = torch.randn(batch_size, in_channels, depth_in, height_in, width_in)
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
