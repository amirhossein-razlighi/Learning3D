import torch
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRasterizer,
    MeshRenderer,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    AlphaCompositor,
    HardPhongShader,
)
from typing import Tuple
from pathlib import Path


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


def get_pointcloud_renderer(
    rendered_image_size: Tuple[int, int] = (512, 512),
    device: torch.device = None,
    radius: float = 0.01,
    bg_colour: Tuple[int, int, int] = (1, 1, 1),
):
    """
    Returns a pointcloud renderer object
    """

    if device is None:
        device = get_device()

    rasterization_settings = PointsRasterizationSettings(
        image_size=rendered_image_size,
        radius=radius,
        points_per_pixel=8,
    )
    rasterizer = PointsRasterizer(raster_settings=rasterization_settings)
    compositor = AlphaCompositor(background_color=bg_colour)
    renderer = PointsRenderer(rasterizer=rasterizer, compositor=compositor)
    return renderer.to(device)


def get_mesh_renderer(
    rendered_image_size: Tuple[int, int] = (512, 512),
    lights: torch.Tensor = None,
    device: torch.device = None,
):
    if device is None:
        device = get_device()

    raster_settings = RasterizationSettings(
        image_size=rendered_image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    rasterizer = MeshRasterizer(raster_settings=raster_settings)
    renderer = MeshRenderer(
        rasterizer=rasterizer,
        shader=HardPhongShader(device=device, lights=lights),
    )

    return renderer.to(device)
