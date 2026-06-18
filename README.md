# From Memorization to Parameter Interference: How Overtraining Experts Harms Model Merging

Official code for the ICML 2026 paper
**From Memorization to Parameter Interference: How Overtraining Experts Harms Model Merging**.

* **Authors:** Stefan Horoi, Guy Wolf, Eugene Belilovsky, Gintare Karolina Dziugaite
* **Paper:** [arXiv:2506.14126](https://arxiv.org/abs/2506.14126)

**Do better expert models always lead to better merged models?** Not always.
Better experts can lead to significantly worse merged models, and how long you
finetune each expert has a major effect on how well it merges. We examine the
phenomenon through the lens of data difficulty, showing that later training
steps are primarily driven by the memorization of a small fraction of difficult
examples, which are predominantly forgotten when merging. We study this across a
range of merging methods, models, and tasks.

This repository provides the training, evaluation, and merging pipeline used in
the paper, supporting **full finetuning** and **LoRA** experts on **T5** and
**BERT**, with checkpoints saved at exponentially spaced training steps so that
merging quality can be measured as a function of expert training length.

## Attribution

This code is derived from the [TIES-Merging](https://github.com/prateeky2806/ties-merging)
repository (Yadav et al., NeurIPS 2023), which itself builds on the
[T-Few / IA3](https://github.com/r-three/t-few) infrastructure
(Copyright © 2022 Salesforce, Inc.). It is distributed under the same
BSD 3-Clause License — see [`LICENSE.txt`](./LICENSE.txt) and [`NOTICE`](./NOTICE).

## Setup

```bash
conda create -n overtrained-merging python=3.8 -y
conda activate overtrained-merging

python -m pip install -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install -U datasets
pip install --upgrade evaluate
```

The Story Cloze dataset must be
[downloaded manually](https://cs.rochester.edu/nlp/rocstories/); point the
`STORY_CLOZE_DIR` environment variable at your local copy before using the
`story_cloze` task.

### Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `MERGING_OUTPUT_DIR` | Root directory for experiment outputs (logs, scores, checkpoints written during training). | `./exp_out` |
| `MERGING_CHECKPOINTS_DIR` | Root directory holding the per-task finetuned checkpoints used for merging (one subfolder per base model, e.g. `t5-base/`). Overridden by an explicit `load_dir=...` argument. | `./checkpoints` |
| `WANDB_ENTITY` | Weights & Biases entity (team/user) for logging. | your default entity |
| `WANDB_MODE` | Set to `disabled` to run without any W&B logging. | unset |
| `HF_HOME` | Relocate the Hugging Face cache. | `~/.cache/huggingface` |
| `STORY_CLOZE_DIR` | Path to a manually downloaded Story Cloze dataset. | unset |
| `GCP_BUCKET` | GCS bucket name for optional checkpoint upload (only used if `should_save_to_gcp=true`). | unset |

## Training experts

Each expert is trained on a single task. Checkpoints are saved at exponentially
spaced steps (`1, 3, 7, …, 2047`) so that merging can be evaluated as a function
of training length. The paper uses the tasks
`paws qasc quartz story_cloze wiki_qa winogrande wsc` with seeds `0 1 2`.

### Full finetuning

```bash
for dataset in paws qasc quartz story_cloze wiki_qa winogrande wsc; do
  for seed in 0 1 2; do
    python src/training.py -c configs/t5_base.json -k \
      project_name=training_seeds_2048 experiment_name="${dataset}_s${seed}" \
      train_dataset=$dataset inference_dataset=$dataset \
      train_dataset_mixture=None inference_dataset_mixture=None \
      num_batches=2048 split=validation seed=$seed
  done
done
```

### LoRA

```bash
for dataset in paws qasc quartz story_cloze wiki_qa winogrande wsc; do
  for seed in 0 1 2; do
    python src/training_lora.py -c configs/t5_base.json -k \
      project_name=training_lora experiment_name="${dataset}_s${seed}" \
      train_dataset=$dataset inference_dataset=$dataset \
      train_dataset_mixture=None inference_dataset_mixture=None \
      num_batches=2048 split=validation seed=$seed lr=0.0005
  done
done
```

BERT experts use `configs/bert.json` in place of `configs/t5_base.json`.

## Evaluating individual experts

Evaluate a single checkpoint at a given training step:

```bash
for dataset in paws qasc quartz story_cloze wiki_qa winogrande wsc; do
  for seed in 0 1 2; do
    for step in 1 3 7 15 31 63 127 255 511 1023 2047; do
      python src/inference.py -c configs/t5_base.json -i $dataset --kwargs \
        checkpoint_to_directly_load_model="${MERGING_OUTPUT_DIR}/training_seeds_2048/t5-base/${dataset}_s${seed}/checkpoints/checkpoint_${step}.pt" \
        split=test project_name=inference_seeds_2048 \
        experiment_name="ckpt_${dataset}_s${seed}_${step}_test"
    done
  done
done
```

## Merging experts

### Supported methods

| Method | `-f` string (single run) | Validation sweep form |
|---|---|---|
| [Mean (simple averaging)](https://arxiv.org/abs/2203.05482) | `basic_mean` | — |
| [Task Arithmetic](https://arxiv.org/abs/2212.04089) | `task-vector_0.3` | `task-vector_linear+0.1+1.01+0.1` |
| [TIES](https://arxiv.org/abs/2306.01708) | `topk20_mass_dis-sum_0.3` | `topk20_mass_dis-sum_linear+0.8+2.51+0.1` |
| [DARE](https://arxiv.org/abs/2311.03099) | `dare0.9_0.3` | `dare0.9_linear+0.1+1.01+0.1` |
| [Iso-C](https://arxiv.org/abs/2502.04959) | `iso-c_3.2` | `iso-c_linear+2.0+6.01+0.2` |
| [Iso-CTS](https://arxiv.org/abs/2502.04959) | `iso-cts0.8_3.2` | `iso-cts0.8_linear+2.0+6.01+0.2` |
| [TSV-M](https://arxiv.org/abs/2412.00081) | `tsv-m_3.2` | `tsv-m_linear+2.0+6.01+0.2` |

The scaling value after the final `_` may be a single float (e.g. `0.3`) or a
linear sweep `linear+start+end+step`. For TIES, `topk20` is the redundancy
threshold, `mass` the sign-election rule, and `dis-sum` the disjoint
aggregation. For DARE, `dare0.9` keeps 90% of the task-vector entries. For
Iso-CTS, `iso-cts0.8` sets the common-subspace fraction.

### Workflow

Merging consists of two stages: a **hyper-parameter sweep on the validation
split**, followed by a **single run on the test split** using the selected
hyper-parameter. `-i`/`-m t5_mixture` selects the mixture of tasks to merge.
`load_dir` points at the directory of trained experts (one subfolder per base
model); it defaults to `MERGING_CHECKPOINTS_DIR` if omitted.

### 1. Validation sweep

```bash
CKPT_DIR=${MERGING_CHECKPOINTS_DIR:-./checkpoints}

python src/merging.py -c configs/t5_base.json -i t5_mixture -m t5_mixture \
  -f task-vector_linear+0.1+1.01+0.1 --kwargs \
  split=validation project_name=merging_ta_valid experiment_name="ckpt_s0_2047" \
  load_dir="${CKPT_DIR}/t5-base" load_seed=0 step=2047
```

Select the best scaling value across the sweep based on the validation scores in
each run's `lambda_*/inference_scores.json`.

### 2. Test run with the chosen hyper-parameter

```bash
python src/merging.py -c configs/t5_base.json -i t5_mixture -m t5_mixture \
  -f task-vector_0.3 --kwargs \
  split=test project_name=merging_ta_test experiment_name="ckpt_s0_2047" \
  load_dir="${CKPT_DIR}/t5-base" load_seed=0 step=2047
```

### Merging LoRA experts

Add `lora="true"` and point `load_dir` at the LoRA training directory:

```bash
# Validation sweep
python src/merging.py -c configs/t5_base.json -i t5_mixture -m t5_mixture \
  -f iso-c_linear+2.0+6.01+0.2 --kwargs \
  split=validation project_name=merging_lora_iso-c_valid experiment_name="ckpt_s0_2047" \
  load_dir="${CKPT_DIR}/t5-base" load_seed=0 step=2047 lora="true"

# Test run with chosen hyper-parameter
python src/merging.py -c configs/t5_base.json -i t5_mixture -m t5_mixture \
  -f iso-c_3.2 --kwargs \
  split=test project_name=merging_lora_iso-c_test experiment_name="ckpt_s0_2047" \
  load_dir="${CKPT_DIR}/t5-base" load_seed=0 step=2047 lora="true"
```

Swap the `-f` string for any method in the [table above](#supported-methods).
BERT merging uses `configs/bert.json`.

## Repository structure

```
configs/         model/training configs (t5_base, t5_large, t5_3b, bert)
src/
  training.py        full finetuning of a single-task expert
  training_lora.py   LoRA finetuning
  training_fast.py   faster training loop variant
  inference.py       evaluation of a checkpoint
  merging.py         merge experts and evaluate (entry point for all 7 methods)
  data/  eval/  model/  train/  utils/   pipeline internals
```

## Citation

```bibtex
@misc{horoi2026memorizationparameterinterferenceovertraining,
      title={From Memorization to Parameter Interference: How Overtraining Experts Harms Model Merging},
      author={Stefan Horoi and Guy Wolf and Eugene Belilovsky and Gintare Karolina Dziugaite},
      year={2026},
      eprint={2506.14126},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.14126},
}
```

If you use the merging methods or the underlying infrastructure, please also
consider citing TIES-Merging (Yadav et al., 2023).

## License

BSD 3-Clause License — see [`LICENSE.txt`](./LICENSE.txt).
