#!/bin/bash

#SBATCH --job-name=docprompt
#SBATCH --nodes=1
#SBATCH --mem=50G
#SBATCH -c 8
#SBATCH --gres=gpu:A6000:1
#SBATCH --time 1-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=yueqis@andrew.cmu.edu

source ~/miniconda3/etc/profile.d/conda.sh
conda activate docprompt

python3 retriever/simcse/run_inference.py \
  --model_name "neulab/docprompting-codet5-python-doc-retriever" \
  --source_file data/task.txt \
  --target_file data/openapi_v2.yaml \
  --source_embed_save_file data/src_embedding \
  --target_embed_save_file data/tgt_embedding \
  --oracle_eval_file data/oracle.json \
  --sim_func cls_distance.cosine \
  --num_layers 12 \
  --save_file data/retrieval_result.json