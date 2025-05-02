import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 1D convolution
conv1d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv1d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_length,
    int kernel_size,
    int stride,
    int dilation,
    bool use_bias) {
    
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= batch_size * out_channels * (input_length - ((kernel_size - 1) * dilation + 1)) / stride + 1) return;

    int batch = out_idx / (out_channels * ((input_length - ((kernel_size - 1) * dilation + 1)) / stride + 1));
    int rem = out_idx % (out_channels * ((input_length - ((kernel_size - 1) * dilation + 1)) / stride + 1));
    int out_channel = rem / ((input_length - ((kernel_size - 1) * dilation + 1)) / stride + 1);
    int out_pos = rem % ((input_length - ((kernel_size - 1) * dilation + 1)) / stride + 1);

    float acc = 0.0f;
    for (int k = 0; k < kernel_size; ++k) {
        int in_pos = out_pos * stride + k * dilation;
        for (int in_channel = 0; in_channel < in_channels; ++in_channel) {
            acc += input[batch * in_channels * input_length + in_channel * input_length + in_pos] *
                   weight[out_channel * in_channels * kernel_size + in_channel * kernel_size + k];
        }
    }

    if (use_bias) {
        acc += bias[out_channel];
    }

    output[out_idx] = acc;
}

torch::Tensor conv1d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                          int stride, int dilation, bool use_bias) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_length = input.size(2);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);

    int output_length = (input_length - ((kernel_size - 1) * dilation + 1)) / stride + 1;
    auto output = torch::zeros({batch_size, out_channels, output_length}, input.options());

    int num_threads = 256;
    int num_blocks = (batch_size * out_channels * output_length + num_threads - 1) / num_threads;

    conv1d_kernel<<<num_blocks, num_threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels, input_length,
        kernel_size, stride, dilation, use_bias);

    return output;
}
"""

conv1d_cpp_source = """
torch::Tensor conv1d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                          int stride, int dilation, bool use_bias);
"""

# Compile the inline CUDA code for 1D convolution
conv1d_cuda_op = load_inline(
    name="conv1d_cuda",
    cpp_sources=conv1d_cpp_source,
    cuda_sources=conv1d_cuda_source,
    functions=["conv1d_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.use_bias = bias

        # Register weights and bias as parameters
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        # Initialize weights and bias
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        if bias:
            nn.init.zeros_(self.bias)

        # Custom CUDA op
        self.conv1d_cuda = conv1d_cuda_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv1d_cuda.conv1d_cuda(
            x, self.weight, self.bias, self.stride, self.dilation, self.use_bias
        )


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a standard 1D convolution operation with asymmetric input and a square kernel, potentially dilated and strided.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 1D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        return self.conv1d(x)

# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
length = 256
stride = 3
dilation = 4

def get_inputs():
    x = torch.randn(batch_size, in_channels, length)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, dilation]


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
