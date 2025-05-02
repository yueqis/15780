import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA kernel for 2D convolution with asymmetric kernel size
conv2d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void conv2d_kernel(
    const float* input, 
    const float* weight, 
    float* output,
    int batch_size, 
    int in_channels, 
    int out_channels,
    int in_height, 
    int in_width,
    int out_height, 
    int out_width,
    int kernel_h, 
    int kernel_w,
    int stride_h, 
    int stride_w,
    int pad_h, 
    int pad_w,
    int dilation_h, 
    int dilation_w
) {
    int oc = blockIdx.z;
    int b = (blockIdx.x >> 5); // Use higher bits for batch
    int h_start = (blockIdx.x & 31) * BLOCK_SIZE; // Use lower bits for height
    int w = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Shared memory for input and weights
    __shared__ float shared_input[3][16][16];
    __shared__ float shared_weight[16][16];
    
    // Initialize output to zero
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (int i = 0; i < 5; ++i) {
            int current_h = h_start + i;
            if (current_h < out_height) {
                output[(b * out_channels + oc) * out_height * out_width + (current_h * out_width + w)] = 0.0f;
            }
        }
    }
    __syncthreads();
    
    // Loop over tiles in the input
    for (int tile = 0; tile < ((in_height + BLOCK_SIZE - 1) / BLOCK_SIZE) * ((in_width + BLOCK_SIZE - 1) / BLOCK_SIZE); ++tile) {
        int tile_h = (tile / ((in_width + BLOCK_SIZE - 1) / BLOCK_SIZE)) * BLOCK_SIZE;
        int tile_w = (tile % ((in_width + BLOCK_SIZE - 1) / BLOCK_SIZE)) * BLOCK_SIZE;
        
        // Load input into shared memory
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int i = 0; i < 5; ++i) {
                int h = h_start + i * 4;
                if (h < in_height && tile_w + threadIdx.x < in_width) {
                    int padded_h = h - pad_h;
                    int padded_w = tile_w + threadIdx.x - pad_w;
                    if (padded_h >= 0 && padded_w >= 0 && padded_h < in_height && padded_w < in_width) {
                        shared_input[ic][h - h_start][threadIdx.x] = input[((b * in_channels + ic) * in_height + padded_h) * in_width + padded_w];
                    } else {
                        shared_input[ic][h - h_start][threadIdx.x] = 0.0f;
                    }
                }
            }
        }
        __syncthreads();
        
        // Load weights into shared memory
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                if (oc < out_channels && threadIdx.x < in_channels) {
                    shared_weight[kh][kw * in_channels + threadIdx.x] = weight[(oc * in_channels * kernel_h * kernel_w) + (threadIdx.x * kernel_h * kernel_w) + (kh * kernel_w) + kw];
                }
            }
        }
        __syncthreads();
        
        // Perform convolution
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int in_h = tile_h + kh * dilation_h;
                    int in_w = tile_w + threadIdx.x + kw * dilation_w;
                    
                    if (in_h >= 0 && in_w >= 0 && in_h < in_height && in_w < in_width) {
                        int out_h = (in_h + pad_h - kh * dilation_h) / stride_h;
                        int out_w = (in_w + pad_w - kw * dilation_w) / stride_w;
                        
                        if (out_h >= 0 && out_w >= 0 && out_h < out_height && out_w < out_width) {
                            atomicAdd(&output[(b * out_channels + oc) * out_height * out_width + (out_h * out_width + out_w)], 
                                      shared_input[ic][in_h - h_start][in_w - tile_w] * shared_weight[kh][kw * in_channels + ic]);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
}

torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight,
                          int stride_h, int stride_w,
                          int pad_h, int pad_w,
                          int dilation_h, int dilation_w) {
    // Input dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    
    // Weight dimensions
    int out_channels = weight.size(0);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    
    // Calculate output dimensions
    int out_height = (in_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_width = (in_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    
    // Create output tensor
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());
    
    // Set grid and block sizes
    dim3 threads(32, 8);  // 32x8 threads per block
    dim3 blocks(((batch_size * ((out_height + BLOCK_SIZE - 1) / BLOCK_SIZE)) * out_width));  // Adjusted block calculation
    
    // Launch kernel
    conv2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_height, in_width,
        out_height, out_width,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w
    );
    
    return output;
}
"""

# C++ interface declaration
conv2d_cpp_source = """
torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight,
                          int stride_h, int stride_w,
                          int pad_h, int pad_w,
                          int dilation_h, int dilation_w);
"""

# Compile the inline CUDA code
conv2d_op = load_inline(
    name="conv2d_op",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_cuda_source,
    functions=["conv2d_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
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

        # Create weight parameter
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *kernel_size)
        )

        # Initialize weight using Kaiming initialization
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")

        self.use_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            self.bias.data.uniform_(-1.0, 1.0)
        else:
            self.register_parameter("bias", None)

        # Register the custom CUDA operator
        self.conv2d_cuda = conv2d_op.conv2d_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Call our custom CUDA implementation of convolution
        result = self.conv2d_cuda(
            x,
            self.weight,
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1],
            self.dilation[0],
            self.dilation[1],
        )

        # Add bias if needed
        if self.use_bias:
            result = result + self.bias.view(1, -1, 1, 1)

        return result


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a standard 2D convolution operation with asymmetric input and kernel sizes.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Tuple of two integers representing the height and width of the convolution kernel.
        stride (tuple, optional): Tuple of two integers representing the stride in the height and width dimensions. Defaults to (1, 1).
        padding (tuple, optional): Tuple of two integers representing the padding in the height and width dimensions. Defaults to (0, 0).
        dilation (tuple, optional): Tuple of two integers representing the dilation in the height and width dimensions. Defaults to (1, 1).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return self.conv2d(x)

# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = (3, 5)  # Asymmetric kernel
height = 256
width = 128  # Asymmetric input dimensions

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
