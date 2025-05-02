import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for transposed convolution
conv_transpose2d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Simplified 2D transposed convolution CUDA kernel (1x1 kernel support only for demonstration)
__global__ void conv_transpose2d_kernel(
    const float* input, 
    const float* weight, 
    float* output,
    int batch_size, int in_channels, int out_channels, int height, int width,
    int kernel_h, int kernel_w, int stride_h, int stride_w,
    int padding_h, int padding_w, int dilation_h, int dilation_w,
    int groups) {

    // Global thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Output dimensions
    int out_h = (height - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_h - 1) + 1;
    int out_w = (width - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_w - 1) + 1;

    // Total elements in output tensor
    int total_elements = batch_size * out_channels * out_h * out_w;
    
    for (int idx = tid; idx < total_elements; idx += gridDim.x * blockDim.x) {
        int n = idx / (out_channels * out_h * out_w);
        int rest = idx % (out_channels * out_h * out_w);
        int c_out = rest / (out_h * out_w);
        int rest2 = rest % (out_h * out_w);
        int h_out = rest2 / out_w;
        int w_out = rest2 % out_w;

        int group = c_out / (out_channels / groups);

        float val = 0.f;
        int c_in_group = in_channels / groups;
        int c_start = group * c_in_group;
        int c_end = c_start + c_in_group;

        for (int c_in = c_start; c_in < c_end; ++c_in) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int h_in = h_out / stride_h - padding_h + kh * dilation_h;
                    int w_in = w_out / stride_w - padding_w + kw * dilation_w;

                    if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                        float w = weight[(c_out * in_channels * kernel_h * kernel_w) + (c_in * kernel_h * kernel_w) + (kh * kernel_w) + kw];
                        float v = input[(n * in_channels * height * width) + (c_in * height * width) + (h_in * width) + w_in];
                        val += v * w;
                    }
                }
            }
        }

        output[idx] = val;
    }
}

torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight,
                                   int kernel_h, int kernel_w,
                                   int stride_h, int stride_w,
                                   int padding_h, int padding_w,
                                   int dilation_h, int dilation_w,
                                   int groups) {
    auto size = input.sizes();
    int batch_size = size[0];
    int in_channels = size[1];
    int height = size[2];
    int width = size[3];

    int out_channels = weight.size(0);

    auto output = torch::zeros({batch_size, out_channels, 
        (height - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_h - 1) + 1,
        (width - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_w - 1) + 1}, 
        input.options());

    const int block_size = 256;
    const int num_blocks = 128; // Fixed number of blocks for simplicity

    conv_transpose2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels, height, width,
        kernel_h, kernel_w, stride_h, stride_w,
        padding_h, padding_w, dilation_h, dilation_w,
        groups);

    return output;
}
"""

conv_transpose2d_cpp_source = """
torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight,
                                   int kernel_h, int kernel_w,
                                   int stride_h, int stride_w,
                                   int padding_h, int padding_w,
                                   int dilation_h, int dilation_w,
                                   int groups);
"""

# Compile the inline CUDA code
conv_transpose2d_op = load_inline(
    name="conv_transpose2d",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_cuda_source,
    functions=["conv_transpose2d_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Custom implementation of transposed convolution using CUDA.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
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

        # Initialize weights like ConvTranspose2d
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *kernel_size)
        )

        if bias:
            self.bias_param = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias_param", None)

        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=0.25)
        if self.bias:
            nn.init.zeros_(self.bias_param)

        # Register custom op
        self.custom_conv = conv_transpose2d_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with custom CUDA-based transposed convolution.
        """
        out = self.custom_conv.conv_transpose2d_cuda(
            x,
            self.weight,
            self.kernel_size[0],
            self.kernel_size[1],
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1],
            self.dilation[0],
            self.dilation[1],
            self.groups,
        )

        if self.bias:
            out = out + self.bias_param.view(1, -1, 1, 1)

        return out


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a 2D transposed convolution operation with asymmetric input, asymmetric kernel, 
    grouped, padded, and dilated.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (height, width).
        stride (tuple, optional): Stride of the convolution (height, width). Defaults to (1, 1).
        padding (tuple, optional): Padding applied to the input (height, width). Defaults to (0, 0).
        dilation (tuple, optional): Spacing between kernel elements (height, width). Defaults to (1, 1).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution.

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
kernel_size = (3, 5)
height = 128
width = 256
stride = (2, 3)
padding = (1, 2)
dilation = (2, 1)
groups = 4

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation, groups]


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
