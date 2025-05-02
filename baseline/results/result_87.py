import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Inline CUDA kernel for pointwise 2D convolution (kernel_size=1)
pointwise_conv2d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Pointwise 2D Convolution: input shape (B, C_in, H, W), weight shape (C_out, C_in, 1, 1), bias optional
__global__ void pointwise_conv2d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int B, int C_in, int C_out, int H, int W) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int num_elements = B * H * W * C_out;
    for (; idx < num_elements; idx += stride) {
        int b = idx / (H * W * C_out);
        int h = (idx / (C_out * W)) % H;
        int w = (idx / C_out) % W;
        int c_out = idx % C_out;

        float sum = 0.0f;
        for (int c_in = 0; c_in < C_in; ++c_in) {
            int input_idx = b * C_in * H * W + c_in * H * W + h * W + w;
            int weight_idx = c_out * C_in + c_in;
            sum += input[input_idx] * weight[weight_idx];
        }

        if (bias != nullptr) {
            sum += bias[c_out];
        }

        output[idx] = sum;
    }
}

torch::Tensor pointwise_conv2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    int B = input.size(0);
    int C_in = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int C_out = weight.size(0);

    auto output = torch::zeros({B, C_out, H, W}, input.options());

    int threads_per_block = 256;
    int num_blocks = (output.numel() + threads_per_block - 1) / threads_per_block;

    const float* bias_data = bias.data_ptr<float>();

    pointwise_conv2d_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_data,
        output.data_ptr<float>(),
        B, C_in, C_out, H, W);

    return output;
}
"""

pointwise_conv2d_cpp_source = """
torch::Tensor pointwise_conv2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the inline CUDA code
pointwise_conv2d = load_inline(
    name="pointwise_conv2d",
    cpp_sources=pointwise_conv2d_cpp_source,
    cuda_sources=pointwise_conv2d_cuda_source,
    functions=["pointwise_conv2d_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias_enabled = bias

        # Define parameters
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, 1, 1))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        # Initialize weights and biases
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        if bias:
            nn.init.zeros_(self.bias)

        # Bind compiled CUDA function
        self.cuda_op = pointwise_conv2d.pointwise_conv2d_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is on GPU
        x = x.cuda()
        weight = self.weight.cuda()
        bias = self.bias.cuda() if self.bias is not None else torch.tensor([]).cuda()

        # Call custom CUDA implementation
        return self.cuda_op(x, weight, bias)


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a pointwise 2D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(Model, self).__init__()
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the pointwise 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        return self.conv1d(x)

# Test code
batch_size = 16
in_channels = 3
out_channels = 64
width = 256
height = 256

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels]


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
