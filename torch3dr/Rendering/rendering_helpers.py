from typing import Tuple
from pathlib import Path

from torch3dr.io import load_obj_mesh
from torch3dr.utils import get_device
from torch3dr.rendering import get_mesh_renderer
import torch
import pytorch3d

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


def render_textured_obj(
    obj_path: Path,
    rendered_img_size: int,
    R_Relative: torch.tensor = torch.eye(3),
    T_Relative: torch.tensor = torch.zeros((1, 3)),
    device: str = None,
    verbose: bool = False,
) -> torch.tensor:
    """
    Render a textured object; Given relative rotation and translation.

    Args:
        obj_path (Path): Path to the .obj file
        rendered_img_size (int): Size of the rendered image
        R_Relative (torch.tensor, optional): Relative Rotation matrix (3x3)
        T_Relative (torch.tensor, optional): Relative Translation vector (1x3)
        device (str, optional): device to use for rendering
        verbose (bool, optional): Print debug information.

    Returns:
        torch.tensor: Rendered Image
    """

    if device is None:
        device = get_device()

    if verbose:
        logger.info(f"Loading the obj file as a mesh: {obj_path}")
    # Load the mesh
    mesh = load_obj_mesh(obj_path, return_type="mesh", device=device)

    R_ = R_Relative @ torch.eye(3, device=device)
    T_ = R_Relative @ torch.tensor([0.0, 0.0, 3.0], device=device) + T_Relative

    if verbose:
        logger.info(f"Initializing Renderer and Cameras and Lights")

    renderer = get_mesh_renderer(rendered_image_size=rendered_img_size, device=device)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R_.unsqueeze(0), T=T_, device=device
    )
    lights = pytorch3d.renderer.PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    if verbose:
        logger.info(f"Rendering the mesh")

    rendered = renderer(mesh, cameras=cameras, lights=lights)
    return rendered[0, ..., :3]
