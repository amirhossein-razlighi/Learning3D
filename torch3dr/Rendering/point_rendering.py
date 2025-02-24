import torch
import pytorch3d
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import PointsRenderer
from typing import Tuple

from torch3dr.utils import *
from .rendering_utils import get_pointcloud_renderer


def render_pointcloud_from_path(
    pointcloud_path: str,
    rendered_img_size: Tuple[int, int] = (256, 256),
    background_color: Tuple[int, int, int] = (1, 1, 1),
    device: str = None,
) -> torch.Tensor:
    """
    Render a point cloud from a file path.
    Args:
      pointcloud_path: Path to the point cloud file.
      rendered_img_size: Size of the rendered image.
      background_color: Background color of the rendered image.
      device: Device to use for rendering.
    Returns:
      Rendered image.
    """

    if device is None:
        device = get_device()

    # Load the point cloud.
    pointcloud = load_data_from_general_file(pointcloud_path)

    local_batch_size = 50
    points_ = pointcloud["verts"][::local_batch_size].unsqueeze(0)
    rgbs_ = pointcloud["rgb"][::local_batch_size].unsqueeze(0)

    pc_obj = Pointclouds(points=points_, features=rgbs_)

    R, T = pytorch3d.renderer.look_at_view_transform(4, 10, 0)

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)

    renderer = get_pointcloud_renderer(
        rendered_image_size=rendered_img_size,
        bg_colour=background_color,
        device=device,
    )

    out = renderer(pc_obj, cameras=cameras)
    out = out[0, ..., :3].cpu().numpy()

    return out
