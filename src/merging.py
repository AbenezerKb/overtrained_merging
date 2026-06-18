import sys, os

print(f"Current working directory: {os.getcwd()}")
sys.path.insert(0, os.getcwd())
# Hugging Face caches to ~/.cache/huggingface by default; set the HF_HOME
# environment variable yourself if you want the cache somewhere else.


import wandb
import argparse
import logging
import torch
import numpy as np
import subprocess
import json

from tqdm import tqdm
from src.data.dataset_mixtures import (
    DATASET_MIXTURES,
    get_dataset_from_argparse,
    get_datasetMixture,
)


from src.train.TrainingConfig import TrainingConfig

from src.model.ModelConfig import ModelConfig
from src.model.load_model import (
    load_model,
    construct_model,
    get_pretrained_model,
    load_checkpoints,
    merge_checkpoints,
    loadCheckpoint_intoModel,
    check_parameterNamesMatch,
)

from src.model.utils import get_parameterCount
from src.model.checkpoint_filepaths import (
    generate_checkpointFilepath,
    generate_MixtureCheckpointFilepath,
)

from src.data.Batcher import Batcher
from src.data.dataset_readers import get_datasetReader
from src.data.dataset_mixtures import get_datasetMixtureReader, DatasetMixtureReader
from src.data.PytorchDataset import PytorchDataset
from src.inference import inference

from src.utils.utils import (
    parse_modelName,
    ParseKwargs,
    get_logger,
    set_seeds,
    setup_wandb_logger,
)
from src.utils.merge_utils import *

logger = logging.getLogger("root")


def resolve_lambda_code(lambda_code):
    if type(lambda_code) is tuple:
        lambda_list = torch.tensor(lambda_code)
    elif "linear+" in lambda_code:
        search_lambda, start, end, step = lambda_code.split("+")
        lambda_list = np.arange(eval(start), eval(end), eval(step))
    elif "mergelist" in lambda_code:
        task_lambdas = lambda_code.split("+")[-1].split(",")
        lambda_list = np.array(task_lambdas).astype(float).tolist()
    else:
        try:
            lambda_value = float(lambda_code)
            lambda_list = np.array([lambda_value])
        except ValueError:
            raise NotImplementedError(f"Unable to decode lambda_code {lambda_code}")
    return lambda_list


def merge_and_evalaute(
    config_toInit,
    merge_function,
    dataset_mixture_to_merge,
    multiple_prompts,
    all_inference_dataset_mixtures,
    # tracker,
    inference_kwargs,
    device,
):
    # Load pretrained model.
    model_config = ModelConfig(
        configDict_toInitializeFrom=config_toInit.get_dict(),
    )
    model, tokenizer, peftParameter_regex = construct_model(
        model_config.pretrained_model,
        model_config.peft_method,
        model_config.max_seq_len,
        device="cpu",
        model_config=model_config,
    )
    print(inference_kwargs)
    logger.info(f"Loading Checkpoints for Merging: {config_toInit.pretrained_model}")
    all_mixing_datasets = get_dataset_from_argparse(dataset_mixture_to_merge)
    dataset_folder = ",".join(d for d in all_mixing_datasets)
    if "bert" in config_toInit.pretrained_model.lower():
        (
            tv_flat_checks,
            flat_ptm,
            flat_ft,
            ft_checks,
            ptm_check,
            merging_tasks,
            dump_dir,
            remove_keys,
            reshape_keys,
            base_model,
            load_dir,
            filepaths,
        ) = load_bert(
            all_mixing_datasets,
            config_toInit.pretrained_model,
            pretrained_state_dict=model.state_dict(),
            load_dir = inference_kwargs.get("load_dir", None),
            load_seed = inference_kwargs.get("load_seed", None),
            step = inference_kwargs.get("step", None),
        )
    elif "t5" in config_toInit.pretrained_model:
        (
            tv_flat_checks,
            flat_ptm,
            flat_ft,
            ft_checks,
            ptm_check,
            merging_tasks,
            dump_dir,
            remove_keys,
            reshape_keys,
            base_model,
            load_dir,
            filepaths,
        ) = load_t5(
            all_mixing_datasets,
            config_toInit.pretrained_model,
            pretrained_state_dict=model.state_dict(),
            load_dir = inference_kwargs.get("load_dir", None),
            load_seed = inference_kwargs.get("load_seed", None),
            step = inference_kwargs.get("step", None),
            lora = inference_kwargs.get("lora", "false")
        )
    else:
        raise ValueError(
            f"Merging not implemented for model: {config_toInit.pretrained_model}"
        )

    # Set model to device
    model.to(device)

    cached_dataset_reader = {}
    if "basic" in merge_function:
        logger.info(f"Performing Basic Merging with {merge_function}")
        merge_type, agg_code = merge_function.split("_")
        merged_checkpoint = basic_merging(agg_code, flat_ft, ft_checks, remove_keys)
        # Delete to clean up CPU memory
        del tv_flat_checks, flat_ptm, flat_ft, ft_checks, ptm_check

        model = loadCheckpoint_intoModel(merged_checkpoint, model)
        # Evaluate
        experiment_dir = os.path.join(
            config_toInit.experiment_dir,
            dataset_folder,
        )
        cached_dataset_reader = inference(
            model,
            tokenizer,
            config_toInit,
            model_config,
            cached_datasetReaders=cached_dataset_reader,
            across_multiplePrompts=multiple_prompts,
            experiment_dir=experiment_dir,
            all_inferenceDatasetMixtures=all_inference_dataset_mixtures,
            inference_kwargs=inference_kwargs,
            device=device,
        )
    elif "task-vector" in merge_function:
        merge_type, lambda_code = merge_function.split("_")
        merged_tv = tv_merging(tv_flat_checks)
        # Delete to clean up CPU memory
        reference_state_dict = ft_checks[0]
        del flat_ft, ft_checks, ptm_check

        lambdas = resolve_lambda_code(lambda_code)
        for lam in lambdas:
            lam = round(lam, 1)
            logger.info(
                f"Performing Task Vector Merging with {merge_function} and Lambda {lam:.1f}"
            )
            merged_check = flat_ptm + lam * merged_tv
            merged_checkpoint = vector_to_state_dict(
                merged_check, reference_state_dict, remove_keys=remove_keys
            )
            model = loadCheckpoint_intoModel(merged_checkpoint, model)
            # Evaluate
            experiment_dir = os.path.join(
                config_toInit.experiment_dir,
                dataset_folder,
                f"lambda_{lam:.1f}",
            )
            cached_dataset_reader = inference(
                model,
                tokenizer,
                config_toInit,
                model_config,
                cached_datasetReaders=cached_dataset_reader,
                across_multiplePrompts=multiple_prompts,
                experiment_dir=experiment_dir,
                all_inferenceDatasetMixtures=all_inference_dataset_mixtures,
                inference_kwargs=inference_kwargs,
                device=device,
            )
    elif "iso-c" in merge_function:
        # Both Iso-C ("iso-c_...") and Iso-CTS ("iso-cts<frac>_...") enter here
        # because "iso-c" is a substring of "iso-cts"; the sub-dispatch below
        # then routes Iso-CTS to its own merging function.
        merge_type, lambda_code = merge_function.split("_")

        non_flat_tvs = [get_non_flat_tv(ptm_check, ft_check) for ft_check in ft_checks]
        if "iso-cts" in merge_type:
            common_space_fraction = float(merge_type[len("iso-cts"):])
            merged_tv = isocts_merging(non_flat_tvs, common_space_fraction)
        else:
            merged_tv = isoc_merging(non_flat_tvs)
        # Delete to clean up CPU memory
        del flat_ft, ft_checks, tv_flat_checks, flat_ptm

        lambdas = resolve_lambda_code(lambda_code)
        for lam in lambdas:
            lam = round(lam, 1)
            logger.info(
                f"Performing {merge_type.upper()} Merging with {merge_function} and Lambda {lam:.1f}"
            )
            merged_checkpoint = add_non_flat_tv(ptm_check, lam, merged_tv, remove_keys=remove_keys)
            model = loadCheckpoint_intoModel(merged_checkpoint, model)
            # Evaluate
            path_parts = [config_toInit.experiment_dir, dataset_folder]
            if "iso-cts" in merge_type:
                path_parts.append(f"csf_{common_space_fraction}")
            path_parts.append(f"lambda_{lam:.1f}")
            experiment_dir = os.path.join(*path_parts)
            cached_dataset_reader = inference(
                model,
                tokenizer,
                config_toInit,
                model_config,
                cached_datasetReaders=cached_dataset_reader,
                across_multiplePrompts=multiple_prompts,
                experiment_dir=experiment_dir,
                all_inferenceDatasetMixtures=all_inference_dataset_mixtures,
                inference_kwargs=inference_kwargs,
                device=device,
            )
    elif "tsv-m" in merge_function:
        merge_type, lambda_code = merge_function.split("_")

        non_flat_tvs = [get_non_flat_tv(ptm_check, ft_check) for ft_check in ft_checks]
        merged_tv = tsvm_merging(non_flat_tvs)
        # Delete to clean up CPU memory
        del flat_ft, ft_checks, tv_flat_checks, flat_ptm

        lambdas = resolve_lambda_code(lambda_code)
        for lam in lambdas:
            lam = round(lam, 1)
            logger.info(
                f"Performing TSV-M Merging with {merge_function} and Lambda {lam:.1f}"
            )
            merged_checkpoint = add_non_flat_tv(ptm_check, lam, merged_tv, remove_keys=remove_keys)
            model = loadCheckpoint_intoModel(merged_checkpoint, model)
            # Evaluate
            experiment_dir = os.path.join(
                config_toInit.experiment_dir,
                dataset_folder,
                f"lambda_{lam:.1f}",
            )
            cached_dataset_reader = inference(
                model,
                tokenizer,
                config_toInit,
                model_config,
                cached_datasetReaders=cached_dataset_reader,
                across_multiplePrompts=multiple_prompts,
                experiment_dir=experiment_dir,
                all_inferenceDatasetMixtures=all_inference_dataset_mixtures,
                inference_kwargs=inference_kwargs,
                device=device,
            )
    elif "dare" in merge_function:
        merge_type, lambda_code = merge_function.split("_")
        keep = float(merge_type[len("dare"):])
        reference_state_dict = ft_checks[0]
        # Delete to clean up CPU memory
        del flat_ft, ft_checks, ptm_check

        lambdas = resolve_lambda_code(lambda_code)
        for lam in lambdas:
            lam = round(lam, 1)
            logger.info(
                f"Performing DARE Merging with {merge_function} (keep={keep}) and Lambda {lam:.1f}"
            )
            merged_tv = dare_merging(tv_flat_checks, keep=keep)
            merged_check = flat_ptm + lam * merged_tv
            merged_checkpoint = vector_to_state_dict(
                merged_check, reference_state_dict, remove_keys=remove_keys
            )
            model = loadCheckpoint_intoModel(merged_checkpoint, model)
            experiment_dir = os.path.join(
                config_toInit.experiment_dir,
                dataset_folder,
                f"keep_{keep}",
                f"lambda_{lam:.1f}",
            )
            cached_dataset_reader = inference(
                model,
                tokenizer,
                config_toInit,
                model_config,
                cached_datasetReaders=cached_dataset_reader,
                across_multiplePrompts=multiple_prompts,
                experiment_dir=experiment_dir,
                all_inferenceDatasetMixtures=all_inference_dataset_mixtures,
                inference_kwargs=inference_kwargs,
                device=device,
            )
    else:
        reset, resolve, merge, lambda_code = merge_function.split("_")
        if "topk" in reset:
            reset_type = "topk"
            reset_thresh = eval(reset[len(reset_type) :])
        elif "std" in reset:
            reset_type = "std"
            reset_thresh = eval(reset[len(reset_type) :])
        elif "nf" in reset:
            reset_type = "nf"
            reset_thresh = eval(reset[len(reset_type) :])
        else:
            reset_type = ""
            reset_thresh = "none"

        merged_tv = merge_methods(
            reset_type,
            tv_flat_checks,
            reset_thresh=reset_thresh,
            resolve_method=resolve,
            merge_func=merge,
        )
        # Delete to clean up CPU memory
        reference_state_dict = ft_checks[0]
        del flat_ft, ft_checks, ptm_check

        lambdas = resolve_lambda_code(lambda_code)
        for lam in lambdas:
            lam = round(lam, 1)
            logger.info(
                f"Performing TIES Merging with {merge_function} and Lambda {lam:.1f}"
            )
            merged_check = flat_ptm + lam * merged_tv
            merged_checkpoint = vector_to_state_dict(
                merged_check, reference_state_dict, remove_keys=remove_keys
            )
            model = loadCheckpoint_intoModel(merged_checkpoint, model)
            # Evaluate
            experiment_dir = os.path.join(
                config_toInit.experiment_dir,
                dataset_folder,
                f"lambda_{lam:.1f}",
            )
            cached_dataset_reader = inference(
                model,
                tokenizer,
                config_toInit,
                model_config,
                cached_datasetReaders=cached_dataset_reader,
                across_multiplePrompts=multiple_prompts,
                experiment_dir=experiment_dir,
                all_inferenceDatasetMixtures=all_inference_dataset_mixtures,
                inference_kwargs=inference_kwargs,
                device=device,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_to_init", type=str, required=True)
    parser.add_argument("-i", "--all_inference_dataset_mixtures", type=str, nargs="*")
    parser.add_argument("-m", "--merging_datasets_or_mixtures", type=str, nargs="*")
    parser.add_argument("--multiple_prompts", action="store_true")
    parser.add_argument(
        "--kwargs",
        nargs="*",
        action=ParseKwargs,
        default={
            "split": "validation",
            "should_save_to_gcp": False,
            "world_size": None,
            "eval_template_idx": -1,
            # None -> fall back to BASIC_INFORMATION[model]["load_dir"] in
            # merge_utils (configurable via MERGING_CHECKPOINTS_DIR).
            "load_dir": None,
            "load_seed": 0,
            "step": None,
            "lora": "false",
        },
    )
    # Supported -f merge_function strings (a lambda/scale follows the final "_",
    # given either as a single float "0.3" or a sweep "linear+start+end+step"):
    #   Mean            : basic_mean            (also basic_median, basic_magnitude)
    #   Task Arithmetic : task-vector_0.3
    #   TIES            : topk20_mass_dis-sum_0.3   ({reset}_{resolve}_{merge}_{lambda})
    #   DARE            : dare0.9_0.3               (dare{keep}_{lambda})
    #   Iso-C           : iso-c_3.2
    #   Iso-CTS         : iso-cts0.8_3.2            (iso-cts{common_space_fraction}_{lambda})
    #   TSV-M           : tsv-m_3.2
    parser.add_argument("-f", "--merge_function", type=str)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_toInit = TrainingConfig(
        config_filepaths=[args.config_to_init], kwargs=args.kwargs, create_expDir=True
    )

    log_config_path = os.path.join(os.getcwd(), "src", "utils", "log_config.json")
    logger = get_logger("log.txt", f"{config_toInit.experiment_dir}/", log_config_path)
    logger.info("Starting Experiments of Model merging")

    command = subprocess.list2cmdline(["python"] + sys.argv)
    logger.info(command)

    merge_and_evalaute(
        config_toInit,
        args.merge_function,
        args.merging_datasets_or_mixtures[0].split(","),
        args.multiple_prompts,
        args.all_inference_dataset_mixtures[0].split(","),
        args.kwargs,
        device,
    )
