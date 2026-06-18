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
    reduce_gatheredOutput,
    is_nodeZero,
    is_distributedSetup,
)


from src.data.Batcher import Batcher
from src.data.dataset_readers import get_datasetReader
from src.data.dataset_mixtures import get_datasetMixtureReader, get_datasetMixture
from src.data.PytorchDataset import PytorchDataset

import copy
from src.utils.utils import safe_makedirs
import math
import torch.optim as optim

logger = logging.getLogger("root")


#-------------------------------------------
# LoRA
#-------------------------------------------
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = self.alpha / self.rank
        
        # LoRA weights
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Initialize with Kaiming uniform
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        # LoRA adaptation
        return self.scaling * (x @ self.lora_A @ self.lora_B)

class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank=4, alpha=1.0):
        super().__init__()
        self.linear = linear_layer
        self.lora = LoRALayer(linear_layer.in_features, linear_layer.out_features, rank=rank, alpha=alpha)
        
    def forward(self, x):
        # Original output + LoRA adaptation
        return self.linear(x) + self.lora(x)

def apply_lora_to_linear_layers(model, rank=4, alpha=1.0):
    """Recursively replace all nn.Linear layers with LoRALinear layers."""
    exclude_keywords = ['lm_head']

    for name, module in model.named_children():
        should_exclude = any(keyword in name for keyword in exclude_keywords)
    
        if isinstance(module, nn.Linear) and not should_exclude:
            setattr(model, name, LoRALinear(module, rank=rank, alpha=alpha))
        else:
            apply_lora_to_linear_layers(module, rank=rank, alpha=alpha)
    return model

def freeze_base_params(model):
    """Freeze all parameters except LoRA parameters."""
    for name, param in model.named_parameters():
        if 'lora_A' not in name and 'lora_B' not in name:
            param.requires_grad = False
    return model

def get_lora_params(model):
    """Get all LoRA parameters."""
    params = []
    names = []
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            params.append(param)
            names.append(name)
    return params, names

def merge_lora_to_linear(model):
    """
    Recursively finds all LoRALinear modules in `model`, folds their A,B adapters
    into the base linear weight, and replaces the LoRALinear with a plain nn.Linear.
    Operates in-place.
    """
    for name, child in list(model.named_children()):
        # if this is a LoRA-wrapped linear...
        if isinstance(child, LoRALinear):
            # original nn.Linear
            linear = child.linear
            # the trained LoRA adapter
            lora = child.lora

            # A: (in_features, rank), B: (rank, out_features)
            # delta_Wᵀ = scaling * (A @ B)  has shape (in_features, out_features)
            # so delta_W = (delta_Wᵀ)ᵀ  matches linear.weight.shape = (out_features, in_features)
            delta_W = (lora.lora_A @ lora.lora_B).T.mul_(lora.scaling)

            # fold it into the weight
            linear.weight.data.add_(delta_W)

            # (if you had biases in your LoRA you’d do the same for linear.bias here)

            # replace the wrapper with the “merged” linear
            setattr(model, name, linear)

        else:
            # recurse into children
            merge_lora_to_linear(child)

    return model


#-------------------------------------------
# Modified methods
#-------------------------------------------
def construct_optimizer(
    model, parameters, optimizer_name, learning_rate, weight_decay
):
    """


    Args:
        model:
        trainableParameter_regex:
        optimizer_name:
        learning_rate:
        weight_decay:

    Returns:

    """
    # if parameters is None:
    #     parameters, names = get_modelParameters(
    #         model, trainableParameter_regex, return_names=True
    #     )
    # logger.info(f"Training the parameters: {names}")

    if optimizer_name.lower() == "adam":
        optimizer = optim.Adam(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.98),
            eps=1e-6,
        )

    elif optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(parameters, lr=learning_rate, weight_decay=weight_decay)

    elif optimizer_name.lower() == "adamw":
        optimizer = optim.AdamW(
            parameters, lr=learning_rate, weight_decay=weight_decay, eps=1e-8
        )

    elif optimizer_name.lower() == "adafactor":
        optimizer = Adafactor(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            decay_rate=0,
            relative_step=False,
        )

    else:
        raise ValueError(f"Optimizer {optimizer_name} not implemented yet ")

    return optimizer


class LoRACheckpointer(Checkpointer):
    # --- helper to save adapters + merged model ---
    def _save_lora_and_merged(self, model, base_path: str):
        # 1) raw checkpoint already written by _save_checkpoint
        # 2) save just the adapter weights
        lora_dict = {k: v for k, v in model.state_dict().items() if 'lora_A' in k or 'lora_B' in k}
        torch.save(
            lora_dict,
            base_path.replace(".pt", "_lora_adapters.pt"),
        )

        # 3) fold adapters into a copy of the model
        merged = copy.deepcopy(model)
        merge_lora_to_linear(merged)
        # 4) save the merged-only state dict
        torch.save(
            merged.state_dict(),
            base_path.replace(".pt", "_merged.pt"),
        )

    def checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        scores,
        batch_idx,
        dont_saveModel=False
    ):
        current_log = self._log_metricAndScores(batch_idx, scores)
        self.numCheckpoints_sinceBestCheckpoint += 1

        if not dont_saveModel:
            # always save the latest training state
            if self.should_saveMostRecentState:
                self._save_trainingState(
                    optimizer,
                    scheduler,
                    batch_idx,
                    os.path.join(
                        self.experiment_dir, "training_state.dict"
                    ),
                )

            if self.should_saveEveryCheckpoint:
                checkpoint_dir = os.path.join(
                    self.experiment_dir, "checkpoints"
                )
                safe_makedirs(checkpoint_dir)

                # 1) regular checkpoint
                ckpt_fp = os.path.join(
                    checkpoint_dir, f"checkpoint_{batch_idx}.pt"
                )
                self._save_checkpoint(model, ckpt_fp)

                # 2) adapters + merged
                self._save_lora_and_merged(model, ckpt_fp)

            else:
                # only save if this is a new best
                if self._is_bestCheckpoint(current_log):
                    best_fp = os.path.join(
                        self.experiment_dir, "best_model.pt"
                    )
                    self._save_checkpoint(model, best_fp)
                    self._save_lora_and_merged(model, best_fp)

        # track best
        if self._is_bestCheckpoint(current_log):
            self._update_bestCheckpoint(current_log)
            self._update_and_save_best_mixing_weights(model, current_log)

        return current_log, self.numCheckpoints_sinceBestCheckpoint



#-------------------------------------------
# Training, standard
#-------------------------------------------
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
    # ——— Add LoRA adapters ———
    model = apply_lora_to_linear_layers(model, rank=training_config.lora_rank, alpha=training_config.lora_alpha)
    model = model.to(device)
    model = freeze_base_params(model)

    logger.info(
        f"Parameter count: {get_parameterCount(model, trainableParameter_regex)}"
    )

    lora_params, lora_names = get_lora_params(model)
    logger.info(f"Training the LoRA parameters: {lora_names}")
    optimizer = construct_optimizer(
        model,
        lora_params,
        training_config.optimizer,
        training_config.lr,
        training_config.weight_decay
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
        checkpointer = LoRACheckpointer(
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
        scaler = torch.cuda.amp.GradScaler(enabled=True)

    for i in tqdm(
        range(
            training_config.num_batches * training_config.gradient_accumulation_factor
        )
    ):
        batch_idx = i // (training_config.gradient_accumulation_factor)
        set_seeds(training_config.seed + batch_idx)

        if batch_idx <= batchIdx_toResumeFrom:
            continue

        model.train()

        train_batch = next(train_iterator)

        if training_config.use_bfloat16_during_training:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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
                if training_config.scheduler is None:
                    scaler.step(optimizer)
                else:
                    scaler.step(scheduler)
                scaler.update()
            else:
                optimizer.step()
                if training_config.scheduler is not None:
                    scheduler.step()

            # Reset optimizer
            optimizer.zero_grad()

            # if (batch_idx + 1) % training_config.checkpoint_frequency == 0:
            if (batch_idx + 1) in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
                checkpoint_scores, cached_datasetReaders = evaluate_checkpoint(
                    model,
                    tokenizer,
                    cached_datasetReaders,
                    evaluation_config,
                    inference_dataset_mixture=training_config.inference_dataset_mixture,
                    batch_idx=batch_idx,
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train(device, None, training_config)
