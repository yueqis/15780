# KERNEL-RAG: Retrieval-Augmented Generation for Writing Efficient GPU Kernels

The RAG framework is under `rag/`. We perform retrieval for 100 Level-1 tasks from KernelBench, and results are stored in `retrieval_dataset.json`.

We evaluate both the baseline prompts and advanced prompts on Qwen3-235B-A22B model. To generate all prompts, run commands below:
- Baseline: `cd baseline/prompt; python3 generate_all_prompts.py`
- Advanced: `cd advanced/prompt; python3 generate_all_prompts_and_images.py`.

The GPU kernels written by LLM are stored in `baseline/results` and `advanced/results` in the form of `result_{task_id}.py`. The Python files are post-processed to be directly run for compiling and measuring performance.
