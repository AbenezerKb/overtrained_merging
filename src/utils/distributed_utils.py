import torch


def reduce_gatheredOutput(gathered_output, reduce_fn=None):
    """
    Reduces the output from multiple devices to have the same format as for a single device.

    Args:
        gathered_output:
        reduce_fn:

    Returns:
    """
    reduced_output = {}

    # Combine output of the values for each key
    for iterate_dict in gathered_output:
        for (k, v) in iterate_dict.items():
            if k in reduced_output:
                reduced_output[k].append(v)
            else:
                reduced_output[k] = [v]

    # Reduce the gathered output at each key
    for (k, batch_ofValues) in reduced_output.items():
        if isinstance(batch_ofValues[0], list):
            reduced_output[k] = [item for sublist in batch_ofValues for item in sublist]
        else:
            reduced_output[k] = [item for item in batch_ofValues]

        if reduce_fn is not None:
            reduced_output[k] = reduce_fn(reduced_output[k])

    return reduced_output


def get_default_device():
    """
    Picks the best available single-process device (cuda > mps > cpu).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_autocast_device_type(device):
    """
    Returns the device_type string to pass to torch.autocast for a device
    (a torch.device, or an int CUDA rank).
    """
    if isinstance(device, torch.device):
        return device.type
    return "cuda"


def is_nodeZero(device):
    """


    Args:
        device:

    Returns:

    """
    if isinstance(device, torch.device) and device.type in ("mps", "cpu"):
        return True
    return device == 0 or device == get_default_device()


def is_distributedSetup(world_size):
    """

    Args:
        world_size:

    Returns:

    """
    return world_size is not None
