import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D convolution with asymmetric kernel
conv3d_custom_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Naive implementation of 3D convolution for demonstration purposes
__global__ void conv3d_custom_kernel(
    const float* input,
    const float* weight,
    float* output,
    int batch_size, int in_channels, int out_channels,
    int input_width, int input_height, int input_depth,
    int output_width, int output_height, int output_depth,
    int kernel_size_w, int kernel_size_h, int kernel_size_d,
    int stride_w, int stride_h, int stride_d,
    int padding_w, int padding_h, int padding_d,
    int dilation_w, int dilation_h, int dilation_d) {
    
    int n = blockIdx.z;
    int c_out = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int w_out = idx % output_width;
    int h_out = (idx / output_width) % output_height;
    int d_out = (idx / (output_width * output_height)) % output_depth;

    float acc = 0.0f;
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kw = 0; kw < kernel_size_w; ++kw) {
            for (int kh = 0; kh < kernel_size_h; ++kh) {
                for (int kd = 0; kd < kernel_size_d; ++kd) {
                    int w_in = w_out * stride_w - padding_w + kw * dilation_w;
                    int h_in = h_out * stride_h - padding_h + kh * dilation_h;
                    int d_in = d_out * stride_d - padding_d + kd * dilation_d;

                    if (w_in >= 0 && w_in < input_width &&
                        h_in >= 0 && h_in < input_height &&
                        d_in >= 0 && d_in < input_depth) {
                        
                        float in_val = input[
                            n * in_channels * input_width * input_height * input_depth +
                            c_in * input_width * input_height * input_depth +
                            w_in * input_height * input_depth +
                            h_in * input_depth +
                            d_in];

                        float w_val = weight[
                            c_out * in_channels * kernel_size_w * kernel_size_h * kernel_size_d +
                            c_in * kernel_size_w * kernel_size_h * kernel_size_d +
                            kw * kernel_size_h * kernel_size_d +
                            kh * kernel_size_d +
                            kd];

                        acc += in_val * w_val;
                    }
                }
            }
        }
    }

    output[
        n * out_channels * output_width * output_height * output_depth +
        c_out * output_width * output_height * output_depth +
        w_out * output_height * output_depth +
        h_out * output_depth +
        d_out] = acc;
}

torch::Tensor conv3d_custom_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride_w, int64_t stride_h, int64_t stride_d,
    int64_t padding_w, int64_t padding_h, int64_t padding_d,
    int64_t dilation_w, int64_t dilation_h, int64_t dilation_d,
    int64_t groups) {

    // We are not using bias for this example
    // We assume groups == 1 for simplicity
    // This is a naive implementation for demonstration purposes only
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_width = input.size(2);
    int input_height = input.size(3);
    int input_depth = input.size(4);

    int out_channels = weight.size(0);
    int kernel_size_w = weight.size(2);
    int kernel_size_h = weight.size(3);
    int kernel_size_d = weight.size(4);

    int output_width = (input_width + 2 * padding_w - kernel_size_w) / stride_w + 1;
    int output_height = (input_height + 2 * padding_h - kernel_size_h) / stride_h + 1;
    int output_depth = (input_depth + 2 * padding_d - kernel_size_d) / stride_d + 1;

    auto output = torch::zeros({batch_size, out_channels, output_width, output_height, output_depth}, input.options());

    dim3 blocks((output_width * output_height * output_depth + 255) / 256, out_channels, batch_size);
    dim3 threads(256);

    conv3d_custom_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_width, input_height, input_depth,
        output_width, output_height, output_depth,
        kernel_size_w, kernel_size_h, kernel_size_d,
        stride_w, stride_h, stride_d,
        padding_w, padding_h, padding_d,
        dilation_w, dilation_h, dilation_d);

    return output;
}
"""

conv3d_custom_cpp_source = """
torch::Tensor conv3d_custom_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride_w, int64_t stride_h, int64_t stride_d,
    int64_t padding_w, int64_t padding_h, int64_t padding_d,
    int64_t dilation_w, int64_t dilation_h, int64_t dilation_d,
    int64_t groups);
"""

# Compile the inline CUDA code for custom 3D convolution
conv3d_custom_op = load_inline(
    name="conv3d_custom",
    cpp_sources=conv3d_custom_cpp_source,
    cuda_sources=conv3d_custom_cuda_source,
    functions=["conv3d_custom_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized version of the Model class that uses a custom CUDA kernel for 3D convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
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
        self.bias = bias

        # Create the convolution weights and bias
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *kernel_size)
        )
        if bias:
            self.bias_param = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias_param", None)

        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        if bias:
            nn.init.zeros_(self.bias_param)

        # Reference to the custom CUDA operator
        self.conv3d_custom = conv3d_custom_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 3D convolution using a custom CUDA kernel.
        """
        return self.conv3d_custom.conv3d_custom_cuda(
            x,
            self.weight,
            self.bias_param if self.bias else torch.zeros(0, device=x.device),
            self.stride,
            self.stride,
            self.stride,
            self.padding,
            self.padding,
            self.padding,
            self.dilation,
            self.dilation,
            self.dilation,
            self.groups,
        )


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a standard 3D convolution operation with a square input and an asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (kernel_width, kernel_height, kernel_depth).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Padding applied to the input. Defaults to 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, width, height, depth).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, width_out, height_out, depth_out).
        """
        return self.conv3d(x)

# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = (3, 5, 7)  # Asymmetric kernel
width = 64
height = 64
depth = 64

def get_inputs():
    x = torch.randn(batch_size, in_channels, width, height, depth)
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
