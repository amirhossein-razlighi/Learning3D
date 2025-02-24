import torch
import numpy as np
import torch
import pickle
from pathlib import Path

from typing import Tuple, Optional, Union


def get_device():
    """
    Checks if GPU is available and returns the device accordingly
    """
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        # else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    return device


def load_data_from_general_file(file_path: str) -> Union[torch.Tensor, dict]:
    """
    Load data from a general file. It inferes the extension of
    the file (.npy, .pkl, .pth) and loads the data accordingly.
    Then, returns the torch tensor.
    Args:
      file_path: Path to the file.
    Returns:
        torch.Tensor object Or a dictionary of torch.Tensor objects.
    """

    path_ = Path(file_path)
    ext = path_.suffix

    if ext == ".npy":
        data = np.load(file_path)
        data = torch.from_numpy(data)
    elif ext == ".npz":
        data_npz = np.load(file_path)
        data = {k: torch.from_numpy(v) for k, v in data_npz.items()}
    elif ext == ".pkl":
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        data = torch.tensor(data)
    elif ext == ".pth":
        data = torch.load(file_path)
    else:
        raise ValueError("Unsupported file extension.")

    return data
