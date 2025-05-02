import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for cosine similarity loss: cosine_similarity + mean(1 - similarity)
cosine_loss_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <math.h>

// Computes cosine similarity and 1 - similarity in a single pass, then sums for mean
__global__ void cosine_similarity_loss_kernel(
    const float* predictions,
    const float* targets,
    const int D,
    const int N,
    float* loss_sum) {

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    float dot = 0.0f;
    float norm_pred = 0.0f;
    float norm_target = 0.0f;

    for (int d = 0; d < D; ++d) {
        int idx = n * D + d;
        float p = predictions[idx];
        float t = targets[idx];
        dot += p * t;
        norm_pred += p * p;
        norm_target += t * t;
    }

    float eps = 1e-8f;
    float denom = sqrt(norm_pred) * sqrt(norm_target) + eps;
    float sim = dot / denom;
    float loss = 1.0f - sim;

    // Use atomic add to accumulate sum on GPU memory
    atomicAdd(loss_sum, loss);
}

torch::Tensor cosine_similarity_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    int N = predictions.size(0);
    int D = predictions.size(1);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto loss_sum = torch::zeros({1}, options);

    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    cosine_similarity_loss_kernel<<<gridSize, blockSize>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        D,
        N,
        loss_sum.data_ptr<float>()
    );

    // Compute mean
    return loss_sum / static_cast<float>(N);
}
"""

cosine_loss_cpp_source = "torch::Tensor cosine_similarity_loss_cuda(torch::Tensor predictions, torch::Tensor targets);"

# Compile the inline CUDA code
cosine_loss_op = load_inline(
    name="cosine_similarity_loss",
    cpp_sources=cosine_loss_cpp_source,
    cuda_sources=cosine_loss_cuda_source,
    functions=["cosine_similarity_loss_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.cosine_loss = cosine_loss_op

    def forward(self, predictions, targets):
        return self.cosine_loss.cosine_similarity_loss_cuda(predictions, targets)


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that computes Cosine Similarity Loss for comparing vectors.

    Parameters:
        None
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets):
        cosine_sim = torch.nn.functional.cosine_similarity(predictions, targets, dim=1)
        return torch.mean(1 - cosine_sim)

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
