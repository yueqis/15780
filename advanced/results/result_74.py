import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for transposed 1D convolution
conv1d_transpose_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv1d_transpose_kernel(
    const float* input, 
    const float* weight, 
    const float* bias, 
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int length,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool use_bias)
{
    // Compute thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Total number of output elements per channel
    int out_length = (length - 1) * stride + kernel_size - 2 * padding;
    
    // Total number of output elements
    int total_output_elements = batch_size * out_channels * out_length;
    
    if (tid >= total_output_elements)
        return;

    // Compute batch index, output channel, and output position
    int pos = tid;
    int out_pos = pos % out_length;
    pos /= out_length;
    int out_channel = pos % out_channels;
    pos /= out_channels;
    int batch_idx = pos;

    // Initialize output to zero
    output[tid] = 0.0f;

    // Loop over input channels
    for (int in_channel = 0; in_channel < in_channels; ++in_channel) {
        // Loop over kernel positions
        for (int k = 0; k < kernel_size; ++k) {
            // Compute input position using transposed convolution formula
            int in_pos = (out_pos + padding - dilation * (k - kernel_size / 2)) / stride;
            
            // Check bounds
            if (in_pos >= 0 && in_pos < length) {
                // Input offset
                int input_offset = ((batch_idx * in_channels + in_channel) * length + in_pos);
                
                // Weight offset: (out_channel, in_channel, kernel_pos)
                int weight_offset = ((out_channel * in_channels + in_channel) * kernel_size + k);
                
                // Accumulate the product of input and weight
                output[tid] += input[input_offset] * weight[weight_offset];
            }
        }
    }

    // Add bias if used
    if (use_bias) {
        output[tid] += bias[out_channel];
    }
}

torch::Tensor conv1d_transpose_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                                   int stride, int padding, int dilation, bool use_bias) {
    // Get tensor dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int length = input.size(2);
    
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);

    // Calculate output size
    int out_length = (length - 1) * stride + kernel_size - 2 * padding;
    
    // Create output tensor
    auto output = torch::zeros({batch_size, out_channels, out_length}, input.options());

    // Launch CUDA kernel
    const int block_size = 256;
    const int num_blocks = (output.numel() + block_size - 1) / block_size;
    
    conv1d_transpose_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        length,
        kernel_size,
        stride,
        padding,
        dilation,
        use_bias
    );

    return output;
}
"""

conv1d_transpose_cpp_source = """
torch::Tensor conv1d_transpose_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    int stride,
    int padding,
    int dilation,
    bool use_bias
);
"""

# Compile the inline CUDA code for transposed 1D convolution
conv1d_transpose_op = load_inline(
    name="conv1d_transpose",
    cpp_sources=conv1d_transpose_cpp_source,
    cuda_sources=conv1d_transpose_cuda_source,
    functions=["conv1d_transpose_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
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
        self.use_bias = bias

        # Register learnable parameters
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        if bias:
            nn.init.zeros_(self.bias)

        # Reference to the custom CUDA operator
        self.conv1d_transpose_cuda = conv1d_transpose_op.conv1d_transpose_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Perform the transposed convolution using our CUDA implementation
        return self.conv1d_transpose_cuda(
            x,
            self.weight,
            self.bias if self.bias is not None else torch.empty(0),
            self.stride,
            self.padding,
            self.dilation,
            self.use_bias,
        )


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a transposed 1D convolution operation with square input and asymmetric kernel, optionally with dilation.

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
        self.conv1d_transpose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        return self.conv1d_transpose(x)

# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 5
length = 256
stride = 1
padding = 0
dilation = 3

def get_inputs():
    x = torch.randn(batch_size, in_channels, length)
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
