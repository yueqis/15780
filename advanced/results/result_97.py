import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Cosine Similarity + Mean Loss
cosine_sim_mean_cuda_code = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void cosine_similarity_mean_kernel(
    const float* predictions, 
    const float* targets, 
    float* output,
    int batch_size, 
    int feature_dim) {
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch_size) return;

    float dot_product = 0.0f;
    float norm_pred = 0.0f;
    float norm_target = 0.0f;

    for (int col = 0; col < feature_dim; ++col) {
        int index = row * feature_dim + col;
        float p = predictions[index];
        float t = targets[index];
        dot_product += p * t;
        norm_pred += p * p;
        norm_target += t * t;
    }

    // Compute norms
    norm_pred = sqrt(norm_pred);
    norm_target = sqrt(norm_target);

    // Avoid division by zero
    float epsilon = 1e-8f;
    float inv_norm_pred = 1.0f / fmaxf(norm_pred, epsilon);
    float inv_norm_target = 1.0f / fmaxf(norm_target, epsilon);

    // Cosine similarity
    float cos_sim = dot_product * inv_norm_pred * inv_norm_target;

    // Use atomics to accumulate mean in case multiple threads write to same output
    atomicAdd(output, (1.0f - cos_sim) / batch_size);
}

torch::Tensor cosine_similarity_mean_cuda(
    torch::Tensor predictions, 
    torch::Tensor targets) {

    int batch_size = predictions.size(0);
    int feature_dim = predictions.size(1);

    auto output = torch::zeros({1}, predictions.options());

    dim3 block(256);
    dim3 grid((batch_size + block.x - 1) / block.x);

    cosine_similarity_mean_kernel<<<grid, block>>>(
        predictions.data_ptr<float>(), 
        targets.data_ptr<float>(), 
        output.data_ptr<float>(),
        batch_size, 
        feature_dim);

    return output;
}
"""

cosine_sim_mean_cpp_code = """
torch::Tensor cosine_similarity_mean_cuda(torch::Tensor predictions, torch::Tensor targets);
"""

# Compile the inline CUDA code
cosine_similarity_mean_op = load_inline(
    name="cosine_similarity_mean",
    cpp_sources=cosine_sim_mean_cpp_code,
    cuda_sources=cosine_sim_mean_cuda_code,
    functions=["cosine_similarity_mean_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.cosine_similarity_mean = cosine_similarity_mean_op

    def forward(self, predictions, targets):
        return self.cosine_similarity_mean.cosine_similarity_mean_cuda(
            predictions, targets
        )


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
