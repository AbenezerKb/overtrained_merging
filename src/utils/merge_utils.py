import sys, os

print(f"Current working directory: {os.getcwd()}")
sys.path.insert(0, os.getcwd())

import json
import os, copy
import logging
import torch
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import OrderedDict
from transformers import AutoModelForSeq2SeqLM
import torch.nn.functional as F

logger = logging.getLogger("root")

# Default directory holding the per-task finetuned checkpoints, with one
# subfolder per base model. Override with the MERGING_CHECKPOINTS_DIR environment
# variable, or by passing load_dir=... on the command line (which takes
# precedence over this default).
CHECKPOINTS_DIR = os.environ.get("MERGING_CHECKPOINTS_DIR", "checkpoints")

BASIC_INFORMATION = {
    "t5-base": {
        "base_model": "t5-base",
        "dump_dir": f"",
        "load_dir": os.path.join(CHECKPOINTS_DIR, "t5-base"),
    },
    "t5-large": {
        "base_model": "t5-large",
        "dump_dir": f"",
        "load_dir": os.path.join(CHECKPOINTS_DIR, "t5-large"),
    },
    "t5-3b": {
        "base_model": "t5-3b",
        "dump_dir": f"",
        "load_dir": os.path.join(CHECKPOINTS_DIR, "t5-3b"),
    },
    "bert-base-uncased": {
        "base_model": "bert-base-uncased",
        "dump_dir": f"",
        "load_dir": os.path.join(CHECKPOINTS_DIR, "bert-base-uncased"),
    },
}


def state_dict_to_vector(state_dict, remove_keys=[]):
    shared_state_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in shared_state_dict:
            del shared_state_dict[key]
    sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
    return torch.nn.utils.parameters_to_vector(
        [value.reshape(-1) for key, value in sorted_shared_state_dict.items()]
    )


def vector_to_state_dict(vector, state_dict, remove_keys=[]):
    # create a reference dict to define the order of the vector
    reference_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in reference_dict:
            del reference_dict[key]
    sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))

    # create a shared state dict using the refence dict
    torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())

    # add back the encoder and decoder embedding weights.
    if "transformer.shared.weight" in sorted_reference_dict:
        for key in remove_keys:
            sorted_reference_dict[key] = sorted_reference_dict[
                "transformer.shared.weight"
            ]
    return sorted_reference_dict


def add_ptm_to_tv(tv_dict, ptm_dict):
    assert set(tv_dict.keys()) == set(
        ptm_dict.keys()
    ), "Differing parameter names in models."
    final_dict = copy.deepcopy(tv_dict)
    for k, v in ptm_dict.items():
        final_dict[k] = tv_dict[k] + v
    return final_dict


def check_parameterNamesMatch(checkpoints):
    parameter_names = set(checkpoints[0].keys())

    if len(checkpoints) >= 2:
        # raise ValueError("Number of models is less than 2.")
        for checkpoint in checkpoints[1:]:
            current_parameterNames = set(checkpoint.keys())
            if current_parameterNames != parameter_names:
                raise ValueError(
                    "Differing parameter names in models. "
                    f"The different parameters are {parameter_names.symmetric_difference(current_parameterNames)}"
                )


def check_state_dicts_equal(state_dict1, state_dict2):
    if set(state_dict1.keys()) != set(state_dict2.keys()):
        return False

    for key in state_dict1.keys():
        if not torch.equal(state_dict1[key], state_dict2[key]):
            return False

    return True


def greater_than_std_mask(tensor, factor, return_mask=False):
    mask = (tensor - tensor.mean(dim=1).unsqueeze(1)).abs() > factor * tensor.std(
        dim=1
    ).unsqueeze(1)
    if return_mask:
        return tensor * mask, mask.float().mean(dim=1), mask
    return tensor * mask, mask.float().mean(dim=1)


def less_than_std_mask(tensor, factor, return_mask=False):
    mask = (tensor - tensor.mean(dim=1).unsqueeze(1)).abs() < factor * tensor.std(
        dim=1
    ).unsqueeze(1)
    if return_mask:
        return tensor * mask, mask.float().mean(dim=1), mask
    return tensor * mask, mask.float().mean(dim=1)


def topk_values_mask(M, K=0.7, return_mask=False):
    if K > 1:
        K /= 100

    original_shape = M.shape
    if M.dim() == 1:
        M = M.unsqueeze(0)

    n, d = M.shape
    k = int(d * K)
    k = d - k  # Keep top k elements instead of bottom k elements

    # Find the k-th smallest element by magnitude for each row
    kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
    # Create a mask tensor with True for the top k elements in each row
    mask = M.abs() >= kth_values
    final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

    if return_mask:
        return M * final_mask, final_mask.float().mean(dim=1), final_mask
    return M * final_mask, final_mask.float().mean(dim=1)


def bottomk_values_mask(M, K=0.7, return_mask=False):
    if K > 1:
        K /= 100

    original_shape = M.shape
    if M.dim() == 1:
        M = M.unsqueeze(0)

    n, d = M.shape
    k = int(d * K)
    # Find the k-th smallest element by magnitude for each row
    kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)

    # Create a mask tensor with True for the bottom k elements in each row
    mask = M.abs() <= kth_values
    final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

    if return_mask:
        return M * final_mask, final_mask.float().mean(dim=1), final_mask
    return M * final_mask, final_mask.float().mean(dim=1)


def topk_mask_preserve_normfrac(T, normfrac=0.9, return_mask=False):
    row_norms = torch.norm(T, p=2, dim=1, keepdim=True)

    # Calculate the proportion of each element's contribution to its row's norm
    proportion = T.abs() ** 2 / row_norms ** 2

    # Sort the proportions and their indices in descending order
    sorted_proportions, sorted_indices = torch.sort(proportion, dim=1, descending=True)

    # Calculate the cumulative sum of proportions
    cumsum_proportions = torch.cumsum(sorted_proportions, dim=1)

    # Find the indices where cumulative sum >= normfrac
    normfrac_mask = cumsum_proportions >= normfrac
    normfrac_indices = torch.argmax(normfrac_mask.float(), dim=1)

    # Create a range tensor to compare with normfrac_indices
    range_tensor = torch.arange(T.size(1)).unsqueeze(0).expand(T.size(0), -1)

    # Create a mask based on the normfrac_indices
    mask = range_tensor <= normfrac_indices.unsqueeze(1)

    # Initialize final_indices with a value that is out of bounds
    final_indices = torch.full_like(sorted_indices, T.size(1) - 1)

    # Use the mask to get the final indices
    final_indices[mask] = sorted_indices[mask]

    # Initialize the mask with zeros
    M = torch.zeros_like(T, dtype=torch.bool)

    # Use the final indices to update the final mask M
    M.scatter_(1, final_indices, True)

    if return_mask:
        return (T * M), M.float().mean(dim=1), M
    else:
        return (T * M), M.float().mean(dim=1)


def sign_agreement_ratio(M):
    positive_count = (M > 0).float().sum(dim=0)
    negative_count = (M < 0).float().sum(dim=0)

    non_zero_values = positive_count + negative_count

    sar = torch.where(
        non_zero_values != 0,
        torch.abs(positive_count - negative_count) / non_zero_values,
        torch.ones_like(non_zero_values),
    )

    return sar


def replace_noise_and_constant(tensor, mask, replace_factor, sign_tensor):
    tensor[mask] = 0

    if replace_factor != 0:
        tensor[~mask] = replace_factor * tensor.std()
        tensor *= sign_tensor

    return tensor


def plot_topk_norms(flat_checkpoints, task_names, topks):
    check_norms = flat_checkpoints.norm(dim=1)

    all_topk_norm = []
    for topk in topks:
        print(topk)
        topk_vector, *_ = topk_values_mask(flat_checkpoints, K=topk)
        topk_norms = topk_vector.norm(dim=1)
        all_topk_norm.append(topk_norms)
    all_topk_norm = torch.vstack(all_topk_norm)
    task_topk_norms = all_topk_norm.T
    normalized_task_topk_norms = task_topk_norms / check_norms.unsqueeze(1)

    plot_rows(
        np.array(topks),
        np.array(normalized_task_topk_norms),
        labels=[f"{task_names[i]}" for i in range(len(flat_checkpoints))],
        title="Norm of top-k% parameters over total norm for each task",
        x_label="top-k%",
        y_label="Norm@k",
    )


def plot_rows(X, Y, labels=None, title=None, x_label=None, y_label=None):
    if X.shape[0] != Y.shape[1]:
        raise ValueError(
            "Length of vector X must match the number of columns in matrix Y."
        )

    if labels is not None and len(labels) != Y.shape[0]:
        raise ValueError("Number of labels must match the number of rows in matrix Y.")

    for i in range(Y.shape[0]):
        plt.plot(X, Y[i], label=labels[i] if labels is not None else None)

    if title is not None:
        plt.title(title)

    if x_label is not None:
        plt.xlabel(x_label)

    if y_label is not None:
        plt.ylabel(y_label)

    if labels is not None:
        plt.legend()

    plt.show()


def plot_row_histograms(M, task_names, bins=None):
    n, d = M.shape
    num_cols = min(
        n, 4
    )  # Adjust this value to change the number of columns in the subplot grid
    num_rows = (n + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))

    for i, row in enumerate(M):
        ax = (
            axes[i // num_cols, i % num_cols]
            if num_rows > 1 or num_cols > 1
            else axes[i]
        )
        ax.hist(row.numpy(), bins=bins)
        ax.set_title(f"{task_names[i]}")
        ax.set_xlabel("Values")
        ax.set_ylabel("Frequency")

    # Remove extra subplots if any
    for j in range(i + 1, num_rows * num_cols):
        fig.delaxes(axes[j // num_cols, j % num_cols])

    plt.tight_layout()
    plt.show()


def resolve_zero_signs(sign_to_mult, method="majority"):
    majority_sign = torch.sign(sign_to_mult.sum())

    if method == "majority":
        sign_to_mult[sign_to_mult == 0] = majority_sign
    elif method == "minority":
        sign_to_mult[sign_to_mult == 0] = -1 * majority_sign
    return sign_to_mult


def normfrac_based_sign(Tensor):
    row_norms = torch.norm(Tensor, dim=1, keepdim=True)
    norm_fracs = (Tensor ** 2) / row_norms ** 2
    return torch.sign(Tensor[norm_fracs.argmax(dim=0), torch.arange(Tensor.shape[1])])


def normmass_based_sign(Tensor):
    row_norms = torch.norm(Tensor, dim=1, keepdim=True)
    norm_fracs = (Tensor ** 2) / row_norms ** 2
    return (Tensor.sign() * norm_fracs.abs()).sum(dim=0).sign()


def resolve_sign(Tensor, resolve_method):
    if resolve_method == "mass":
        sign_to_mult = torch.sign(Tensor.sum(dim=0))
    elif resolve_method == "normfrac":
        sign_to_mult = normfrac_based_sign(Tensor)
    elif resolve_method == "normmass":
        sign_to_mult = normmass_based_sign(Tensor)
    else:
        raise ValueError(f"Sign resolve method {resolve_method} is not defined.")
    sign_to_mult = resolve_zero_signs(sign_to_mult, "majority")
    return sign_to_mult


def disjoint_merge(Tensor, merge_func, sign_to_mult):

    merge_func = merge_func.split("-")[-1]

    # If sign is provided then we select the corresponding entries and aggregate.
    if sign_to_mult is not None:
        rows_to_keep = torch.where(
            sign_to_mult.unsqueeze(0) > 0, Tensor > 0, Tensor < 0
        )
        selected_entries = Tensor * rows_to_keep
    # Else we select all non-zero entries and aggregate.
    else:
        rows_to_keep = Tensor != 0
        selected_entries = Tensor * rows_to_keep

    if merge_func == "mean":
        non_zero_counts = (selected_entries != 0).sum(dim=0).float()
        disjoint_aggs = torch.sum(selected_entries, dim=0) / torch.clamp(
            non_zero_counts, min=1
        )
    elif merge_func == "sum":
        disjoint_aggs = torch.sum(selected_entries, dim=0)
    elif merge_func == "max":
        disjoint_aggs = selected_entries.abs().max(dim=0)[0]
        disjoint_aggs *= sign_to_mult
    else:
        raise ValueError(f"Merge method {merge_func} is not defined.")

    return disjoint_aggs


def aggregate(T, agg_type, final_signs, dim=0):
    if agg_type == "mean":
        result = torch.mean(T, dim=dim)
    elif agg_type == "sum":
        result = torch.sum(T, dim=dim)
    elif agg_type == "median":
        result = torch.median(T, dim=dim)[0]
    elif agg_type == "magnitude":
        max_indices = T.abs().argmax(dim=0)
        result = T[max_indices, torch.arange(T.shape[1])]
    else:
        raise ValueError("Invalid agg_type: %s" % agg_type)

    if final_signs is not None:
        # print(final_signs)
        result = result.abs() * final_signs

    return result


def basic_merging(merge_func, flat_checks, sd_checks, remove_keys):
    """ "Basic aggregation of the delta checks"""
    all_checks = flat_checks.clone()
    merged_check = aggregate(all_checks, merge_func, final_signs=None)

    final_sd = vector_to_state_dict(merged_check, sd_checks[0], remove_keys=remove_keys)
    return final_sd


def tv_merging(tv_flat_checks):
    """Merging by creating and scaling Task Vectors"""
    all_checks = tv_flat_checks.clone()
    tv_merged_check = aggregate(all_checks, "sum", final_signs=None)
    return tv_merged_check


def dare_merging(tv_flat_checks, keep, rescale=True, generator=None):
    """DARE: per-task Bernoulli mask + rescale, then sum across tasks."""
    if keep > 1:
        keep = keep / 100
    assert 0.0 < keep <= 1.0
    all_checks = tv_flat_checks.clone()
    mask = torch.bernoulli(
        torch.full_like(all_checks, fill_value=keep), generator=generator
    )
    pruned = all_checks * mask
    if rescale:
        pruned = pruned / keep
    return aggregate(pruned, "sum", final_signs=None)


def get_non_flat_tv(ptm_check, ft_check):
    '''get state dict that preserves parameter sizes'''
    # load_t5 already checks that parameters match
    tv = {}
    for key in ptm_check:
        tv[key] = ft_check[key] - ptm_check[key]
    return tv


def add_non_flat_tv(ptm_check, lam, merged_tv, remove_keys=[]):
    tv = {}
    for key in ptm_check:
        if key in remove_keys:
            continue
        tv[key] = ptm_check[key] + lam * merged_tv[key]

    if "transformer.shared.weight" in tv:
        for key in remove_keys:
            tv[key] = tv["transformer.shared.weight"]

    return tv


def isoc_merging(task_vectors):
    '''Iso-C merging method'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        new_vector = {}
        for key in task_vectors[0]:
            orig_device = task_vectors[0][key].device
            tvs = [task_vector[key].to(device) for task_vector in task_vectors]
            new_vector[key] = sum(tvs) / len(tvs)

            if len(task_vectors[0][key].shape) == 2:
                new_vector[key] *= len(tvs)
                U, S, V = torch.linalg.svd(new_vector[key], full_matrices=False)
                S_mean = torch.ones_like(S) * S.mean()

                new_vector[key] = torch.linalg.multi_dot(
                    (
                        U,
                        torch.diag(S_mean),
                        V,
                    )
                )

            new_vector[key] = new_vector[key].to(orig_device)

    return new_vector


@torch.no_grad()
def isocts_merging(task_vectors, common_space_fraction):
    '''Iso-CTS merging method: common + task-specific SVD spaces'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_tasks = len(task_vectors)
    new_vector = {}

    for key in task_vectors[0]:
        shape_ = task_vectors[0][key].shape
        orig_device = task_vectors[0][key].device
        is_2d_matrix = len(shape_) == 2

        if not is_2d_matrix:
            # Running average for non-2D parameters
            for i, task_vector in enumerate(task_vectors):
                vec = task_vector[key].to(device)
                if i == 0:
                    new_vector[key] = vec.clone()
                else:
                    new_vector[key] += (vec - new_vector[key]) / (i + 1)
            new_vector[key] = new_vector[key].to(orig_device)
            continue

        # 2D matrix: common + task-specific space decomposition
        combined_w = sum(task_vector[key].to(device) for task_vector in task_vectors)

        # Calculate common space size, ensuring task-specific space is evenly divisible
        min_dim = min(shape_)
        common_space_index_s = int(min_dim * common_space_fraction)
        task_specific_total = round((min_dim - common_space_index_s) / num_tasks) * num_tasks
        common_space_index_s = min_dim - task_specific_total

        # SVD of combined task vectors for common space
        u, s, v = torch.linalg.svd(combined_w, full_matrices=False)
        common_space_u = u[:, :common_space_index_s]
        common_space_s = s[:common_space_index_s]
        common_space_v = v[:common_space_index_s, :]

        # Calculate task-specific spaces
        n_dims_per_task = int((min_dim - common_space_index_s) / num_tasks)
        for i, task_vector in enumerate(task_vectors):
            w = task_vector[key].to(device)

            # Remove common space projection
            w_ts = w - common_space_u @ common_space_u.T @ w
            u_ts, s_ts, v_ts = torch.linalg.svd(w_ts, full_matrices=False)

            if i == 0:
                combined_space_u = torch.zeros_like(u_ts, device=device)
                combined_space_s = torch.zeros_like(s_ts, device=device)
                combined_space_v = torch.zeros_like(v_ts, device=device)

            combined_space_u[:, i * n_dims_per_task : (i + 1) * n_dims_per_task] = u_ts[:, :n_dims_per_task]
            combined_space_s[i * n_dims_per_task : (i + 1) * n_dims_per_task] = s_ts[:n_dims_per_task]
            combined_space_v[i * n_dims_per_task : (i + 1) * n_dims_per_task, :] = v_ts[:n_dims_per_task, :]

        # Insert common space after task-specific spaces
        cs_start = num_tasks * n_dims_per_task
        cs_end = cs_start + common_space_index_s
        combined_space_u[:, cs_start:cs_end] = common_space_u
        combined_space_s[cs_start:cs_end] = common_space_s
        combined_space_v[cs_start:cs_end, :] = common_space_v

        # Orthogonalize U and V
        u_u, _, v_u = torch.linalg.svd(combined_space_u, full_matrices=False)
        u_v, _, v_v = torch.linalg.svd(combined_space_v, full_matrices=False)
        combined_space_u = u_u @ v_u
        combined_space_v = u_v @ v_v

        # Isotropic singular values
        combined_space_s = torch.ones_like(combined_space_s) * combined_space_s.mean()

        new_vector[key] = torch.linalg.multi_dot(
            (
                combined_space_u,
                torch.diag(combined_space_s),
                combined_space_v,
            )
        ).to(orig_device)

    return new_vector


@torch.no_grad()
def tsvm_merging(task_vectors):
    '''TSV-M merging method: per-task SVD reduction, concatenation, orthogonalization'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_tasks = len(task_vectors)
    sv_reduction = 1 / num_tasks
    new_vector = {}

    for key in task_vectors[0]:
        shape_ = task_vectors[0][key].shape
        orig_device = task_vectors[0][key].device
        is_2d_matrix = len(shape_) == 2

        if not is_2d_matrix:
            # Running average for non-2D parameters
            for i, task_vector in enumerate(task_vectors):
                vec = task_vector[key].to(device)
                if i == 0:
                    new_vector[key] = vec.clone()
                else:
                    new_vector[key] += (vec - new_vector[key]) / (i + 1)
            new_vector[key] = new_vector[key].to(orig_device)
            continue

        # 2D matrix: per-task SVD reduction and concatenation
        for i, task_vector in enumerate(task_vectors):
            vec = task_vector[key].to(device)
            u, s, v = torch.linalg.svd(vec, full_matrices=False)

            if i == 0:
                sum_u = torch.zeros_like(u, device=device)
                sum_s = torch.zeros_like(s, device=device)
                sum_v = torch.zeros_like(v, device=device)

            reduced_index_s = int(s.shape[0] * sv_reduction)
            sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[:, :reduced_index_s]
            sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[:reduced_index_s]
            sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[:reduced_index_s, :]

        # Orthogonalize U and V
        u_u, _, v_u = torch.linalg.svd(sum_u, full_matrices=False)
        u_v, _, v_v = torch.linalg.svd(sum_v, full_matrices=False)

        new_vector[key] = torch.linalg.multi_dot(
            (
                u_u,
                v_u,
                torch.diag(sum_s),
                u_v,
                v_v,
            )
        ).to(orig_device)

    return new_vector


def merge_methods(
    reset_type,
    flat_task_checks,
    reset_thresh=None,
    resolve_method=None,
    merge_func="",
):
    all_checks = flat_task_checks.clone()

    if "nf" in reset_type and reset_thresh != "none":
        logger.info(f"Pruning: {reset_type} {reset_thresh}")
        updated_checks, *_ = topk_mask_preserve_normfrac(
            all_checks, reset_thresh, return_mask=False
        )
    elif "topk" in reset_type and reset_thresh != "none":
        logger.info(f"Pruning: {reset_type} {reset_thresh}")
        updated_checks, *_ = topk_values_mask(
            all_checks, K=reset_thresh, return_mask=False
        )
    elif "std" in reset_type and reset_thresh != "none":
        logger.info(f"Pruning: {reset_type} {reset_thresh}")
        updated_checks, *_ = greater_than_std_mask(
            all_checks, reset_thresh, return_mask=False
        )
    else:
        logger.info("Not removing NOISE")
        updated_checks = all_checks

    if resolve_method != "none":
        logger.info(f"RESOLVING SIGN: {resolve_method}")
        final_signs = resolve_sign(updated_checks, resolve_method)
        assert final_signs is not None
    else:
        logger.info("Not RESOLVING SIGN")
        final_signs = None

    if "dis" in merge_func:
        logger.info(f"Disjoint AGGREGATION: {merge_func}")
        merged_tv = disjoint_merge(updated_checks, merge_func, final_signs)
    else:
        logger.info(f"Basic AGGREGATION: {merge_func}")
        merged_tv = aggregate(updated_checks, merge_func, final_signs)

    return merged_tv


def load_t5(merging_tasks, model_key, pretrained_state_dict, load_dir=None, load_seed=None, step=None, lora='false'):

    dump_dir = BASIC_INFORMATION[model_key]["dump_dir"]
    base_model = BASIC_INFORMATION[model_key]["base_model"]
    # load_dir = BASIC_INFORMATION[model_key]["load_dir"]

    # load the finetune and pretrained checkpoints
    if load_dir is None and load_seed is None and step is None:
        load_dir = BASIC_INFORMATION[model_key]["load_dir"]
        filepaths = [os.path.join(load_dir, task, "best_model.pt") for task in merging_tasks]
    elif step is None:
        filepaths = [os.path.join(load_dir, f"{task}_s{load_seed}", f"best_model.pt") for task in merging_tasks]
    elif load_seed is None:
        filepaths = [os.path.join(load_dir, task, f"checkpoints/checkpoint_{step}.pt") for task in merging_tasks]
    elif lora == "true":
        filepaths = [os.path.join(load_dir, f"{task}_s{load_seed}", f"checkpoints/checkpoint_{step}_merged.pt") for task in merging_tasks]
    else:
        filepaths = [os.path.join(load_dir, f"{task}_s{load_seed}", f"checkpoints/checkpoint_{step}.pt") for task in merging_tasks]

    if lora == 'true':
        ft_checks = [torch.load(fp, map_location="cpu") for fp in filepaths]
    else:
        ft_checks = [torch.load(fp) for fp in filepaths]
    ptm_check = pretrained_state_dict

    # check if all checkpoints have the same paramters.
    check_parameterNamesMatch(ft_checks + [ptm_check])
    remove_keys = [
        "transformer.encoder.embed_tokens.weight",
        "transformer.decoder.embed_tokens.weight",
    ]
    reshape_keys = ["transformer.shared.weight", "transformer.lm_head.weight"]
    for key in remove_keys + reshape_keys:
        ptm_check[key] = ptm_check[key][: len(ft_checks[0][key]), :]

    logger.info(f"Flattening out Checkpoints")
    flat_ft = torch.vstack(
        [state_dict_to_vector(check, remove_keys) for check in ft_checks]
    )
    flat_ptm = state_dict_to_vector(ptm_check, remove_keys)
    # if lora == 'true':
    #     flat_ft  = flat_ft.cpu()
    tv_flat_checks = flat_ft - flat_ptm

    # check if the vectorized state dicts can be converted back to the original state dicts.
    assert check_state_dicts_equal(
        vector_to_state_dict(flat_ptm, ft_checks[0], remove_keys), ptm_check
    )
    assert check_state_dicts_equal(
        vector_to_state_dict(flat_ptm, ptm_check, remove_keys), ptm_check
    )
    assert all(
        [
            check_state_dicts_equal(
                vector_to_state_dict(flat_ft[i], ptm_check, remove_keys), ft_checks[i]
            )
            for i in range(len(ft_checks))
        ]
    )

    logger.info(
        f"Pretrained Model: {base_model}\tMean:{flat_ptm.mean():.6f}\tSTD: {flat_ptm.std():.6f}"
    )
    logger.info(f"Merging tasks: {merging_tasks}")
    logger.info(
        f"Finetune checkpoint \nMean:{flat_ft.mean(dim=1)}\nSTD: {flat_ft.std(dim=1)}"
    )
    logger.info(
        f"Task Vector Finetune checkpoint\nMean:{tv_flat_checks.mean(dim=1)}\nSTD: {tv_flat_checks.std(dim=1)}"
    )

    return (
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
    )


def load_bert(merging_tasks, model_key, pretrained_state_dict, load_dir=None, load_seed=None, step=None):
    """
    Load BERT checkpoints for merging.
    Similar to load_t5 but adapted for BERT's encoder-only architecture.
    """
    dump_dir = BASIC_INFORMATION[model_key]["dump_dir"]
    base_model = BASIC_INFORMATION[model_key]["base_model"]

    # Construct file paths (same logic as T5)
    if load_dir is None and load_seed is None and step is None:
        load_dir = BASIC_INFORMATION[model_key]["load_dir"]
        filepaths = [os.path.join(load_dir, task, "best_model.pt") for task in merging_tasks]
    elif step is None:
        filepaths = [os.path.join(load_dir, f"{task}_s{load_seed}", f"best_model.pt") for task in merging_tasks]
    elif load_seed is None:
        filepaths = [os.path.join(load_dir, task, f"checkpoints/checkpoint_{step}.pt") for task in merging_tasks]
    else:
        filepaths = [os.path.join(load_dir, f"{task}_s{load_seed}", f"checkpoints/checkpoint_{step}.pt") for task in merging_tasks]

    ft_checks = [torch.load(fp) for fp in filepaths]
    ptm_check = pretrained_state_dict

    # Check if all checkpoints have the same parameters
    check_parameterNamesMatch(ft_checks + [ptm_check])
    
    # BERT doesn't have encoder/decoder embedding tying like T5
    # We include all parameters in the merging
    remove_keys = []
    reshape_keys = []

    logger.info(f"Flattening out Checkpoints")
    flat_ft = torch.vstack(
        [state_dict_to_vector(check, remove_keys) for check in ft_checks]
    )
    flat_ptm = state_dict_to_vector(ptm_check, remove_keys)
    tv_flat_checks = flat_ft - flat_ptm

    # Verify vectorization is reversible
    assert check_state_dicts_equal(
        vector_to_state_dict(flat_ptm, ft_checks[0], remove_keys), ptm_check
    )
    assert check_state_dicts_equal(
        vector_to_state_dict(flat_ptm, ptm_check, remove_keys), ptm_check
    )
    assert all(
        [
            check_state_dicts_equal(
                vector_to_state_dict(flat_ft[i], ptm_check, remove_keys), ft_checks[i]
            )
            for i in range(len(ft_checks))
        ]
    )

    logger.info(
        f"Pretrained Model: {base_model}\tMean:{flat_ptm.mean():.6f}\tSTD: {flat_ptm.std():.6f}"
    )
    logger.info(f"Merging tasks: {merging_tasks}")
    logger.info(
        f"Finetune checkpoint \nMean:{flat_ft.mean(dim=1)}\nSTD: {flat_ft.std(dim=1)}"
    )
    logger.info(
        f"Task Vector Finetune checkpoint\nMean:{tv_flat_checks.mean(dim=1)}\nSTD: {tv_flat_checks.std(dim=1)}"
    )

    return (
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
    )
