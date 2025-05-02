import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Inline CUDA kernel for fused MSE computation
mse_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mse_kernel(const float* predictions, const float* targets, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float sdata[];
    
    if (idx < size) {
        float diff = predictions[idx] - targets[idx];
        sdata[threadIdx.x] = diff * diff;
    } else {
        sdata[threadIdx.x] = 0.0f;
    }

    __syncthreads();

    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Final warp reduce
    if (threadIdx.x < 32) {
        sdata[threadIdx.x] += sdata[threadIdx.x + 32];
        __syncwarp();
        sdata[threadIdx.x] += sdata[threadIdx.x + 16];
        __syncwarp();
        sdata[threadIdx.x] += sdata[threadIdx.x + 8];
        __syncwarp();
        sdata[threadIdx.x] += sdata[threadIdx.x + 4];
        __syncwarp();
        sdata[threadIdx.x] += sdata[threadIdx.x + 2];
        __syncwarp();
        sdata[threadIdx.x] += sdata[threadIdx.x + 1];
        __syncwarp();
    }

    if (threadIdx.x == 0) {
        atomicAdd(output, sdata[0] / size);
    }
}

torch::Tensor mse_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto size = predictions.numel();
    auto output = torch::zeros({}, predictions.options());

    const int block_size = 512;
    const int num_blocks = 1;  // Single block since we're doing a global reduction

    size_t smem_size = block_size * sizeof(float);

    mse_kernel<<<num_blocks, block_size, smem_size>>>(predictions.data_ptr<float>(),
                                                     targets.data_ptr<float>(),
                                                     output.data_ptr<float>(),
                                                     size);

    return output;
}
"""

mse_cpp_source = (
    "torch::Tensor mse_cuda(torch::Tensor predictions, torch::Tensor targets);"
)

# Compile the inline CUDA code
mse_op = load_inline(
    name="mse_op",
    cpp_sources=mse_cpp_source,
    cuda_sources=mse_cuda_source,
    functions=["mse_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.mse_op = mse_op

    def forward(self, predictions, targets):
        return self.mse_op.mse_cuda(predictions, targets)


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that computes the Mean Squared Error loss for regression tasks.

    Parameters:
        None
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets):
        return torch.mean((predictions - targets) ** 2)

batch_size = 128
input_shape = (4096, )
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape), torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return []



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
