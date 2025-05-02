import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Inline CUDA kernel for fused softmax and cross entropy loss
cross_entropy_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void cross_entropy_kernel(const float* predictions, const int64_t* targets, float* loss, int num_classes, int batch_size) {
    int b = blockIdx.x;
    int t = threadIdx.x;

    extern __shared__ float s_data[];

    // Softmax computation
    float max_val = -1e20f;
    for (int c = t; c < num_classes; c += blockDim.x) {
        max_val = fmaxf(max_val, predictions[b * num_classes + c]);
    }

    // Reduce max across threads
    for (int i = blockDim.x >> 1; i > 0; i >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFF, max_val, i);
        max_val = fmaxf(max_val, other);
    }
    
    // Compute exps and sum
    float sum_exp = 0.0f;
    for (int c = t; c < num_classes; c += blockDim.x) {
        float exp_val = expf(predictions[b * num_classes + c] - max_val);
        s_data[c] = exp_val;
        sum_exp += exp_val;
    }

    // Reduce sum_exp across threads
    for (int i = blockDim.x >> 1; i > 0; i >>= 1) {
        sum_exp += __shfl_down_sync(0xFFFFFFFF, sum_exp, i);
    }

    // Normalize and compute log probability for true class
    float log_prob = 0.0f;
    int target_class = targets[b];
    for (int c = t; c < num_classes; c += blockDim.x) {
        float prob = s_data[c] / sum_exp;
        if (c == target_class) {
            log_prob = logf(prob);
        }
    }

    // Reduce log_prob across threads
    for (int i = blockDim.x >> 1; i > 0; i >>= 1) {
        log_prob += __shfl_down_sync(0xFFFFFFFF, log_prob, i);
    }

    if (t == 0) {
        loss[b] = -log_prob;
    }
}

torch::Tensor cross_entropy_cuda(torch::Tensor predictions, torch::Tensor targets) {
    int batch_size = predictions.size(0);
    int num_classes = predictions.size(1);

    auto loss = torch::zeros({batch_size}, predictions.options());

    dim3 grid(batch_size);
    dim3 block(256);  // Can be adjusted based on your GPU architecture

    cross_entropy_kernel<<<grid, block, num_classes * sizeof(float)>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        loss.data_ptr<float>(),
        num_classes,
        batch_size
    );

    return loss.mean();
}
"""

cross_entropy_cpp_source = """
torch::Tensor cross_entropy_cuda(torch::Tensor predictions, torch::Tensor targets);
"""

# Compile the inline CUDA code
cross_entropy_op = load_inline(
    name="cross_entropy",
    cpp_sources=cross_entropy_cpp_source,
    cuda_sources=cross_entropy_cuda_source,
    functions=["cross_entropy_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernel for Cross Entropy Loss.
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        self.cross_entropy = cross_entropy_op

    def forward(self, predictions, targets):
        return self.cross_entropy.cross_entropy_cuda(predictions, targets)


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that computes Cross Entropy Loss for multi-class classification tasks.

    Parameters:
        None
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets):
        return torch.nn.functional.cross_entropy(predictions, targets)

batch_size = 4096
num_classes = 10
input_shape = (num_classes, )  # Output for each class
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape), torch.randint(0, num_classes, (batch_size,))]

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
