import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Conv3D + ReLU + Bias
conv3d_relu_bias_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

__global__ void conv3d_relu_bias_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int depth,
    int height,
    int width,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int stride_d,
    int stride_h,
    int stride_w,
    int pad_d,
    int pad_h,
    int pad_w,
    int dilation_d,
    int dilation_h,
    int dilation_w,
    int groups,
    int out_depth,
    int out_height,
    int out_width) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * out_depth * out_height * out_width) return;

    int b = idx / (out_channels * out_depth * out_height * out_width);
    int rest = idx % (out_channels * out_depth * out_height * out_width);
    int c_out = rest / (out_depth * out_height * out_width);
    rest = rest % (out_depth * out_height * out_width);
    int t = rest / (out_height * out_width);
    int h = (rest / out_width) % out_height;
    int w = rest % out_width;

    int d_start = t * stride_d - pad_d;
    int h_start = h * stride_h - pad_h;
    int w_start = w * stride_w - pad_w;

    float sum = 0.0f;
    int g = c_out % groups;
    int group_in_channels = in_channels / groups;
    int group_out_channels = out_channels / groups;

    for (int kd = 0; kd < kernel_d; ++kd) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int d = d_start + kd * dilation_d;
                int m = h_start + kh * dilation_h;
                int n = w_start + kw * dilation_w;

                if (d >= 0 && d < depth && m >= 0 && n >= 0 && m < height && n < width) {
                    for (int c_in = 0; c_in < group_in_channels; ++c_in) {
                        float input_val = input[b * in_channels * depth * height * width +
                                                (g * group_in_channels + c_in) * depth * height * width +
                                                d * height * width +
                                                m * width +
                                                n];

                        float weight_val = weight[c_out * in_channels * kernel_d * kernel_h * kernel_w +
                                                   (g * group_in_channels + c_in) * kernel_d * kernel_h * kernel_w +
                                                   kd * kernel_h * kernel_w +
                                                   kh * kernel_w +
                                                   kw];

                        sum += input_val * weight_val;
                    }
                }
            }
        }
    }

    // Add bias
    sum += bias[c_out];

    // Apply ReLU
    output[idx] = max(sum, 0.0f);
}

torch::Tensor conv3d_relu_bias_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int batch_size,
    int in_channels,
    int out_channels,
    int depth,
    int height,
    int width,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int stride_d,
    int stride_h,
    int stride_w,
    int pad_d,
    int pad_h,
    int pad_w,
    int dilation_d,
    int dilation_h,
    int dilation_w,
    int groups,
    int out_depth,
    int out_height,
    int out_width) {

    auto output = torch::zeros({batch_size, out_channels, out_depth, out_height, out_width}, input.options());

    int num_elements = batch_size * out_channels * out_depth * out_height * out_width;
    int blocks = (num_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    conv3d_relu_bias_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        depth,
        height,
        width,
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        pad_d,
        pad_h,
        pad_w,
        dilation_d,
        dilation_h,
        dilation_w,
        groups,
        out_depth,
        out_height,
        out_width);

    cudaDeviceSynchronize();

    return output;
}
"""

conv3d_relu_bias_cpp_source = """
torch::Tensor conv3d_relu_bias_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int batch_size,
    int in_channels,
    int out_channels,
    int depth,
    int height,
    int width,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int stride_d,
    int stride_h,
    int stride_w,
    int pad_d,
    int pad_h,
    int pad_w,
    int dilation_d,
    int dilation_h,
    int dilation_w,
    int groups,
    int out_depth,
    int out_height,
    int out_width);
"""

# Compile the inline CUDA code
conv3d_relu_bias_op = load_inline(
    name="conv3d_relu_bias",
    cpp_sources=conv3d_relu_bias_cpp_source,
    cuda_sources=conv3d_relu_bias_source,
    functions=["conv3d_relu_bias_cuda"],
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
        dilation: tuple = (1, 1, 1),
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias

        # Register weights and bias as parameters
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        # Initialize weights and bias
        nn.init.kaiming_normal_(self.weight)
        if bias:
            nn.init.constant_(self.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        _, _, depth, height, width = x.size()

        # Calculate output dimensions
        out_depth = (
            depth
            + 2 * self.padding[0]
            - self.dilation[0] * (self.kernel_size[0] - 1)
            - 1
        ) // self.stride[0] + 1
        out_height = (
            height
            + 2 * self.padding[1]
            - self.dilation[1] * (self.kernel_size[1] - 1)
            - 1
        ) // self.stride[1] + 1
        out_width = (
            width
            + 2 * self.padding[2]
            - self.dilation[2] * (self.kernel_size[2] - 1)
            - 1
        ) // self.stride[2] + 1

        return conv3d_relu_bias_op.conv3d_relu_bias_cuda(
            x.contiguous(),
            self.weight.contiguous(),
            (
                self.bias.contiguous()
                if self.bias is not None
                else torch.zeros(self.out_channels, device="cuda")
            ),
            batch_size,
            self.in_channels,
            self.out_channels,
            depth,
            height,
            width,
            self.kernel_size[0],
            self.kernel_size[1],
            self.kernel_size[2],
            self.stride[0],
            self.stride[1],
            self.stride[2],
            self.padding[0],
            self.padding[1],
            self.padding[2],
            self.dilation[0],
            self.dilation[1],
            self.dilation[2],
            self.groups,
            out_depth,
            out_height,
            out_width,
        )


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a standard 3D convolution operation with asymmetric input and kernel sizes.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel in the form (kernel_size_d, kernel_size_h, kernel_size_w).
        stride (tuple, optional): Stride of the convolution in the form (stride_d, stride_h, stride_w). Defaults to (1, 1, 1).
        padding (tuple, optional): Padding applied to the input in the form (padding_d, padding_h, padding_w). Defaults to (0, 0, 0).
        dilation (tuple, optional): Spacing between kernel elements in the form (dilation_d, dilation_h, dilation_w). Defaults to (1, 1, 1).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), dilation: tuple = (1, 1, 1), groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        return self.conv3d(x)

# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = (3, 5, 7)  # Asymmetric kernel size
depth = 16
height = 256
width = 256

def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, height, width)
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
