```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for 2D Average Pooling
avg_pool_2d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool_2d_kernel(
    const float* input,
    float* output,
    int batch_size, int channels, int height, int width,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int out_h, int out_w) {

    int n = blockIdx.z;
    int c = blockIdx.y;
    int ow = blockIdx.x % out_w;
    int oh = blockIdx.x / out_w;

    int iw_start = ow * stride_w - pad_w;
    int ih_start = oh * stride_h - pad_h;

    float sum = 0.0f;
    int count = 0;

    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            int ih = ih_start + kh;
            int iw = iw_start + kw;

            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                sum += input[n * channels * height * width + c * height * width + ih * width + iw];
                count++;
            }
        }
    }

    output[n * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow] = sum / count;
}

torch::Tensor avg_pool_2d_cuda(torch::Tensor input, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w) {
    auto input_sizes = input.sizes();
    int batch_size = input_sizes[0];
    int channels = input_sizes[1];
    int height = input_sizes[2];
    int width = input_sizes[3];

    int out_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;

    auto output = torch::zeros({batch_size, channels, out_h, out_w}, input.options());

    dim3 grid(out_w * out_h, channels, batch_size);
    int block_size = 256;

    avg_pool_2d_kernel<<<grid, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, height, width,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        out_h, out_w);

    return output;
}
"""

avg_pool_2d_cpp_source = """
torch::Tensor avg_pool_2d_cuda(torch::Tensor input, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w);
"""

# Compile the inline CUDA code for custom AvgPool2d
custom_avg_pool = load_inline(
    name="custom_avg_pool",
    cpp_sources=avg_pool_2d_cpp_source,
    cuda_sources=avg_pool_2d_cuda_source,
    functions=["avg_pool_2d_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA kernel for 2D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the optimized Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to None (same as kernel_size).
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.avg_pool_2d_cuda = custom_avg_pool.avg_pool_2d_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized 2D Average Pooling to the input tensor using a custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        return self.avg_pool_2d_cuda(x, self.kernel_size, self.kernel_size, self.stride, self.stride, self.padding, self.padding)
```