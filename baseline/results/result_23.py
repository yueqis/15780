```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Inline CUDA kernel for online softmax (single-pass, numerically stable)
softmax_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void softmax_kernel(float* input, float* output, int rows, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int stride = gridDim.x;
    int col = tid;

    // Pointers to the current row
    float* row_input = input + row * cols;
    float* row_output = output + row * cols;

    // Find max element in the row for numerical stability
    float max_val = -FLT_MAX;
    while (col < cols) {
        max_val = fmaxf(row_input[col], max_val);
        col += blockDim.x;
    }

    // Broadcast max_val across threads using shuffle or shared memory
    __shared__ float shared_max;
    if (threadIdx.x == 0) {
        shared_max = max_val;
    }
    __syncthreads();
    max_val = shared_max;

    // Compute sum of exponentials
    float sum_exp = 0.0f;
    col = threadIdx.x;
    while (col < cols) {
        float exp_val = expf(row_input[col] - max_val);
        sum_exp += exp_val;
        col += blockDim.x;
    }

    // Reduce sum across threads
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        sum_exp += __shfl_down_sync(0xffffffff, sum_exp, s);
    }

    // Normalize and write outputs
    col = threadIdx.x;
    while (col < cols) {
        row_output[col] = expf(row_input[col] - max_val) / sum_exp;
        col += blockDim.x;
    }
}

torch::Tensor softmax_cuda(torch::Tensor input) {
    int rows = input.size(0);
    int cols = input.size(1);

    auto output = torch::zeros_like(input);

    dim3 blocks(rows);
    dim3 threads(1024);

    softmax_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);

    return output;
}
"""

softmax_cpp_source = "torch::Tensor softmax_cuda(torch::Tensor input);"

# Compile inline CUDA code
softmax_op = load_inline(
    name="softmax_cuda",
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_cuda_source,
    functions=["softmax_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softmax_op = softmax_op
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax_op.softmax_cuda(x)
```