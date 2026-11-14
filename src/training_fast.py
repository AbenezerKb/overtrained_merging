import sys, os

print(f"Current working directory: {os.getcwd()}")
sys.path.insert(0, os.getcwd())
# Set the HF_HOME environment variable yourself to relocate the Hugging Face cache.


import torch
import argparse
import logging
from tqdm import tqdm

from collections import OrderedDict

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.nn as nn

from src.model.Checkpointer import Checkpointer
from src.model.load_model import load_model
from src.model.ModelConfig import ModelConfig
from src.model.utils import get_parameterCount

from src.train.TrainingConfig import TrainingConfig
from src.train.utils import (
    load_checkpointToResumeFrom,
    construct_optimizer,
    construct_scheduler,
)

from src.eval.EvaluationConfig import EvaluationConfig
from src.eval.MultiEvaluationConfig import MultiEvaluationConfig
from src.eval.scores import get_averageScores, concatenate_scores, extact_score
from src.eval.evaluate import evaluate_multipleConfigs

from src.utils.utils import (
    group_by,
    map_forDictionaries,
    deep_update,
    ParseKwargs,
    set_seeds,
    get_average,
)
from src.utils.distributed_utils import (
    get_default_device,
    get_autocast_device_type,
    reduce_gatheredOutput,
    is_nodeZero,
    is_distributedSetup,
)


from src.data.Batcher import Batcher
from src.data.dataset_readers import get_datasetReader
from src.data.dataset_mixtures import get_datasetMixtureReader, get_datasetMixture
from src.data.PytorchDataset import PytorchDataset

logger = logging.getLogger("root")
torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True)


from contextlib import nullcontext

def _unwrap_ddp(m):
    return m.module if hasattr(m, "module") else m

def _find_hf_submodule(m):
    # Look for a submodule with a HF config (has model_type, generate, etc.)
    for name, sub in _unwrap_ddp(m).named_modules():
        cfg = getattr(sub, "config", None)
        if cfg is not None and hasattr(cfg, "model_type"):
            return sub
    return None

def enable_train_speedups(model, logger):
    base = _unwrap_ddp(model)
    hf = _find_hf_submodule(base)
    if hf is None:
        logger.warning("No HF submodule with `.config` found; not applying toggles.")
        logger.info(f"Top-level model type: {type(base)}; consider exposing `.model` or `.hf_model` in your wrapper.")
        return

    # 1) Disable KV cache for training (reenable in eval if you want gen speed)
    if hasattr(hf, "config"):
        hf.config.use_cache = False
        logger.info(f"[speedups] use_cache=False on {type(hf).__name__}")

    # 2) Gradient checkpointing: prefer official API, else set flags on stacks
    if hasattr(hf, "gradient_checkpointing_enable"):
        hf.gradient_checkpointing_enable()
        logger.info(f"[speedups] Enabled gradient checkpointing via .gradient_checkpointing_enable()")
    else:
        # Fallback for T5-like models where stacks expose the flag
        enc = getattr(hf, "encoder", None)
        dec = getattr(hf, "decoder", None)
        did = False
        if enc is not None and hasattr(enc, "gradient_checkpointing"):
            enc.gradient_checkpointing = True; did = True
        if dec is not None and hasattr(dec, "gradient_checkpointing"):
            dec.gradient_checkpointing = True; did = True
        logger.info(f"[speedups] Gradient checkpointing fallback set on stacks: {did}")

    # 3) Prefer SDPA/Flash attention if supported; otherwise harmless
    if hasattr(hf, "config") and hasattr(hf.config, "attn_implementation"):
        hf.config.attn_implementation = "sdpa"
        logger.info(f"[speedups] attn_implementation='sdpa'")
    else:
        logger.info(f"[speedups] attn_implementation not in config; relying on torch SDPA backend toggles")

    # Global SDPA kernel prefs (safe no-ops if unsupported shapes)
    torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True)


def evaluate_checkpoint(
    model,
    tokenizer,
    cached_datasetReaders,
    evaluation_config,
    inference_dataset_mixture,
    batch_idx,
    should_evalTrain,
    should_evalValidation,
    device,
):
    """

    Args:
        model:
        evaluation_batchers:
        evaluation_config:
        inference_dataset_mixture:
        batch_idx:
        should_evalTrain:
        should_evalValidation:
        device:

    Returns:

    """
    logger.info(f"Evaluating checkpoint")

    batch_predictionDir = os.path.join(
        evaluation_config.prediction_dir, f"batch_{batch_idx}"
    )

    fields_toIterateOver = []
    fields_toUpdate = {"prediction_dir": batch_predictionDir}

    """
    Compute arguments for evaluating various splits
    """
    splits_toEvaluate = []
    if should_evalTrain:
        splits_toEvaluate.append("train")
    if should_evalValidation:
        splits_toEvaluate.append("validation")

    if len(splits_toEvaluate) > 1:
        fields_toIterateOver.append("split")
        fields_toUpdate["split"] = splits_toEvaluate

    if len(splits_toEvaluate) == 0:
        raise ValueError("No splits to evaluate")

    """
    Compute arguments for evaluating dataset mixture
    """
    if inference_dataset_mixture is not None:
        fields_toIterateOver.append("inference_dataset")
        fields_toUpdate["inference_dataset"] = get_datasetMixture(
            inference_dataset_mixture
        )

    multiEvaluation_config = MultiEvaluationConfig(
        fields_toIterateOver=fields_toIterateOver,
        values_toIterateOver=None,
        configDict_toInitializeFrom=evaluation_config.get_dict(),
        fields_toUpdate=fields_toUpdate,
    )

    multiple_configAndScores, cached_datasetReaders = evaluate_multipleConfigs(
        model, tokenizer, cached_datasetReaders, multiEvaluation_config, device
    )
    if is_nodeZero(device):
        groupScores_bySplit = group_by(
            multiple_configAndScores, lambda x: x["config"]["split"]
        )

        if inference_dataset_mixture is not None:
            averageScore_perSplit = map_forDictionaries(
                my_dict=groupScores_bySplit, map_fn=get_averageScores
            )
            # Since the scores to concatenate are of different datasets, the returned dictionary
            # will show the datset for each score
            concatenatedScores_perSplit = map_forDictionaries(
                my_dict=groupScores_bySplit, map_fn=concatenate_scores
            )
            checkpoint_scores = deep_update(
                concatenatedScores_perSplit, averageScore_perSplit
            )
        else:
            checkpoint_scores = map_forDictionaries(
                my_dict=groupScores_bySplit, map_fn=extact_score
            )

        if "validation" in checkpoint_scores:
            score_toSelectCheckpoint = checkpoint_scores["validation"]["average"]
        elif "test" in checkpoint_scores:
            score_toSelectCheckpoint = checkpoint_scores["test"]["average"]
        else:
            score_toSelectCheckpoint = checkpoint_scores["train"]["average"]

        checkpoint_scores["score_to_select_checkpoint"] = score_toSelectCheckpoint
    else:
        checkpoint_scores = None

    return checkpoint_scores, cached_datasetReaders


def train(device, world_size, training_config):

    if is_distributedSetup(training_config.world_size):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12346"
        torch.cuda.set_device(device)
        dist.init_process_group(
            "nccl", rank=device, world_size=training_config.world_size
        )

    set_seeds(training_config.seed)

    prediction_dir = os.path.join(training_config.experiment_dir, "predictions")
    evaluation_config = EvaluationConfig(
        configDict_toInitializeFrom=training_config.get_dict(),
        fields_toUpdate={"prediction_dir": prediction_dir},
    )

    model_config = ModelConfig(configDict_toInitializeFrom=training_config.get_dict())
    model, tokenizer, trainableParameter_regex, _ = load_model(
        model_config, cached_models={}, device=device
    )

    enable_train_speedups(model, logger)

    logger.info(
        f"Parameter count: {get_parameterCount(model, trainableParameter_regex)}"
    )

    optimizer = construct_optimizer(
        model,
        trainableParameter_regex,
        training_config.optimizer,
        training_config.lr,
        training_config.weight_decay,
    )

    scheduler = None
    if training_config.scheduler is not None:
        scheduler = construct_scheduler(
            optimizer,
            training_config.scheduler,
            training_config.num_batches,
            training_config.warmup_ratio,
        )

    if training_config.resume_checkpoint_filepath is not None:
        (
            model,
            optimizer,
            scheduler,
            batchIdx_toResumeFrom,
            current_bestScore,
        ) = load_checkpointToResumeFrom(
            training_config.resume_checkpoint_filepath, model, optimizer, scheduler
        )

    else:
        batchIdx_toResumeFrom = 0
        current_bestScore = 0

    if is_distributedSetup(training_config.world_size):
        model = DistributedDataParallel(
            model, device_ids=[device], output_device=device
        )

    dataset_kwargs = {
        "few_shot_random_seed": evaluation_config.few_shot_random_seed,
        "num_val_samples": evaluation_config.num_val_samples,
        "max_datapoints_per_dataset_without_templates": training_config.max_datapoints_per_dataset_without_templates,
    }
    if training_config.train_dataset_mixture is not None:

        dataset_reader, cached_datasetReaders = get_datasetMixtureReader(
            training_config.train_dataset_mixture,
            training_config.max_datapoints_per_dataset,
            dataset_kwargs,
        )

    else:
        dataset_reader = get_datasetReader(
            training_config.train_dataset, dataset_kwargs
        )
        cached_datasetReaders = {training_config.train_dataset: dataset_reader}

    createPytorchDataset_fn = lambda dataset: PytorchDataset(dataset, tokenizer, device)
    batcher = Batcher(
        dataset_reader,
        createPytorchDataset_fn,
        train_batchSize=training_config.train_batch_size,
        eval_batchSize=evaluation_config.eval_batch_size,
        world_size=evaluation_config.world_size,
        device=device,
    )

    train_iterator = batcher.get_trainBatches(
        "train", training_config.train_template_idx
    )

    if is_nodeZero(device):
        checkpointer = Checkpointer(
            trainableParameter_regex,
            training_config.experiment_dir,
            training_config.should_save_most_recent_state,
            training_config.should_save_every_checkpoint,
            training_config.world_size,
            training_config.should_save_to_gcp,
            training_config.gradient_accumulation_factor,
            current_bestScore,
        )

    if training_config.should_eval_at_beginning:
        logger.info(f"Evaluating before training")

        checkpoint_scores, cached_datasetReaders = evaluate_checkpoint(
            model,
            tokenizer,
            cached_datasetReaders,
            evaluation_config,
            inference_dataset_mixture=training_config.inference_dataset_mixture,
            batch_idx=0,
            should_evalTrain=training_config.should_eval_train,
            should_evalValidation=training_config.should_eval_validation,
            device=device,
        )

        if is_nodeZero(device):
            checkpointer.checkpoint(
                model, optimizer, scheduler, checkpoint_scores, 0, dont_saveModel=True
            )

    if training_config.use_bfloat16_during_training:
        # bf16 does not need loss scaling; GradScaler is only for cuda + fp16
        # (the autocast dtype here is always bf16, so it is a pass-through).
        scaler = torch.amp.GradScaler(
            get_autocast_device_type(device),
            enabled=False,
        )

    for i in tqdm(
        range(
            training_config.num_batches * training_config.gradient_accumulation_factor
        )
    ):
        batch_idx = i // (training_config.gradient_accumulation_factor)
        # set_seeds(training_config.seed + batch_idx)

        if batch_idx <= batchIdx_toResumeFrom:
            continue

        model.train()

        train_batch = next(train_iterator)

        if training_config.use_bfloat16_during_training:
            with torch.autocast(
                device_type=get_autocast_device_type(device),
                dtype=torch.bfloat16,
            ):
                loss, current_metrics = model(train_batch)
                loss = loss / training_config.gradient_accumulation_factor
            scaler.scale(loss).backward()
        else:
            loss, current_metrics = model(train_batch)
            loss = loss / training_config.gradient_accumulation_factor
            loss.backward()

        if is_distributedSetup(training_config.world_size):
            gathered_currentMetrics = [{}] * training_config.world_size
            dist.gather_object(
                current_metrics,
                gathered_currentMetrics if is_nodeZero(device) else None,
                dst=0,
            )

            if is_nodeZero(device):
                current_metrics = reduce_gatheredOutput(
                    gathered_currentMetrics, get_average
                )

        if is_nodeZero(device):
            checkpointer.update_runningSumOfMetrics(current_metrics)

        if (i + 1) % training_config.gradient_accumulation_factor == 0:
            # Clip norm of gradient
            if training_config.norm_to_clip_gradient is not None:
                # Unscale gradient if using bfloat16 so clipping can be correct magnitude
                if training_config.use_bfloat16_during_training:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), training_config.norm_to_clip_gradient
                )

            # Take a gradient step
            if training_config.use_bfloat16_during_training:
                scaler.step(optimizer)
                scaler.update()
                if training_config.scheduler is not None:
                    scheduler.step()
            else:
                optimizer.step()
                if training_config.scheduler is not None:
                    scheduler.step()

            # Reset optimizer
            optimizer.zero_grad(set_to_none=True)

            if (
                (training_config.early_stopping
                and (batch_idx + 1) % training_config.checkpoint_frequency == 0)
                or
                (not training_config.early_stopping
                and (batch_idx + 1) in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
            ):
                checkpoint_scores, cached_datasetReaders = evaluate_checkpoint(
                    model,
                    tokenizer,
                    cached_datasetReaders,
                    evaluation_config,
                    inference_dataset_mixture=training_config.inference_dataset_mixture,
                    batch_idx=batch_idx + (1 if not training_config.early_stopping else 0),
                    should_evalTrain=training_config.should_eval_train,
                    should_evalValidation=training_config.should_eval_validation,
                    device=device,
                )

                if is_nodeZero(device):
                    (
                        current_log,
                        numCheckpoints_sinceBestCheckpoint,
                    ) = checkpointer.checkpoint(
                        model, optimizer, scheduler, checkpoint_scores, batch_idx
                    )

                    logger.info(f"Finished {batch_idx} batches with log {current_log}")
                    if training_config.early_stopping:
                        if (
                            numCheckpoints_sinceBestCheckpoint
                            >= training_config.early_stopping_num_checkpoints_without_improvement
                        ):
                            if is_distributedSetup(training_config.world_size):
                                dist.destroy_process_group()
                            return

    if is_distributedSetup(training_config.world_size):
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config_filepaths", action="store", type=str, nargs="*", required=True
    )
    parser.add_argument("-d", "--debug_mode", action="store_true")
    parser.add_argument("-k", "--kwargs", nargs="*", action=ParseKwargs, default={})
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger.info("Starting training")

    training_config = TrainingConfig(args.config_filepaths, args.kwargs)

    if training_config.world_size is not None:
        mp.spawn(
            train,
            args=(training_config.world_size, training_config),
            nprocs=training_config.world_size,
        )
    else:
        device = get_default_device()
        train(device, None, training_config)
