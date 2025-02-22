import torch


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
