import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for transposed convolution (very simplified version, only supports specific cases)
transposed_conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Simplified 2D transposed convolution kernel (only stride=1, padding=0, output_padding=0, groups=1)
__global__ void transposed_conv2d_kernel(
    const float* input, const float* weight, float* output,
    int batch_size, int in_channels, int out_channels,
    int height_in, int width_in, int kernel_size) {
    
    int n = blockIdx.z;
    int c_out = blockIdx.y;
    int c_in = threadIdx.z;
    
    extern __shared__ float shared_input[];
    float* input_cache = shared_input;
    
    // Calculate output dimensions
    int height_out = height_in + (kernel_size - 1);
    int width_out = width_in + (kernel_size - 1);
    
    // Loop over spatial dimensions
    for (int h = threadIdx.x; h < height_out; h += blockDim.x) {
        for (int w = threadIdx.y; w < width_out; w += blockDim.y) {
            float val = 0.0f;
            
            // Apply convolution weights
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int h_in = h - kh;
                    int w_in = w - kw;
                    
                    if (h_in >= 0 && w_in >= 0 && h_in < height_in && w_in < width_in) {
                        int input_idx = n * in_channels * height_in * width_in +
                                        c_in * height_in * width_in +
                                        h_in * width_in + w_in;
                        
                        int weight_idx = c_out * in_channels * kernel_size * kernel_size +
                                         c_in * kernel_size * kernel_size +
                                         kh * kernel_size + kw;
                        
                        val += input[input_idx] * weight[weight_idx];
                    }
                }
            }
            
            // Accumulate result
            int output_idx = n * out_channels * height_out * width_out +
                             c_out * height_out * width_out +
                             h * width_out + w;
                             
            atomicAdd(&output[output_idx], val);
        }
    }
}

torch::Tensor transposed_conv2d_cuda(
    torch::Tensor input, torch::Tensor weight,
    int batch_size, int in_channels, int out_channels,
    int height_in, int width_in, int kernel_size) {
    
    // Calculate output dimensions
    int height_out = height_in + (kernel_size - 1);
    int width_out = width_in + (kernel_size - 1);
    
    // Create output tensor
    auto output = torch::zeros({batch_size, out_channels, height_out, width_out}, input.options());
    
    // Set up grid and block dimensions
    dim3 threads(8, 8, 1);  // Reduced thread dimensions for simplicity
    dim3 blocks(1, out_channels, batch_size);  // (x: unused, y: out_channels, z: batch_size)
    
    // Launch kernel
    size_t shared_mem_size = in_channels * height_in * width_in * sizeof(float);
    transposed_conv2d_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, in_channels, out_channels, height_in, width_in, kernel_size);
    
    return output;
}
"""

transposed_conv2d_cpp_source = """
torch::Tensor transposed_conv2d_cuda(
    torch::Tensor input, torch::Tensor weight,
    int batch_size, int in_channels, int out_channels,
    int height_in, int width_in, int kernel_size);
"""

# Compile the inline CUDA code
transposed_conv2d = load_inline(
    name="transposed_conv2d",
    cpp_sources=transposed_conv2d_cpp_source,
    cuda_sources=transposed_conv2d_source,
    functions=["transposed_conv2d_cuda"],
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
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()

        # Only support case where stride == 1 and padding == 0 and output_padding == 0 and groups == 1
        assert stride == 1, "Only stride=1 is supported in custom kernel"
        assert padding == 0, "Only padding=0 is supported in custom kernel"
        assert (
            output_padding == 0
        ), "Only output_padding=0 is supported in custom kernel"
        assert groups == 1, "Only groups=1 is supported in custom kernel"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias

        # Register weight as a parameter
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        )

        # Custom CUDA function
        self.transposed_conv2d = transposed_conv2d

        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, mode="fan_in", nonlinearity="leaky_relu")

        if bias:
            self.bias_param = nn.Parameter(torch.Tensor(out_channels))
            self.bias_param.data.zero_()
        else:
            self.register_parameter("bias_param", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height_in, width_in = x.size()

        # Perform transposed convolution using custom CUDA kernel
        output = self.transposed_conv2d.transposed_conv2d_cuda(
            x,
            self.weight,
            batch_size,
            self.in_channels,
            self.out_channels,
            height_in,
            width_in,
            self.kernel_size,
        )

        # Add bias if enabled
        if self.bias:
            output = output + self.bias_param.view(1, self.out_channels, 1, 1)

        return output


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a transposed 2D convolution with asymmetric input and a square kernel.

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
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return self.conv_transpose2d(x)

# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
height_in = 128
width_in = 256

def get_inputs():
    x = torch.randn(batch_size, in_channels, height_in, width_in)
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
