from .rendering_utils import *
import pytorch3d
import imageio
import torch
from pathlib import Path
import click
import numpy as np
from tqdm.auto import tqdm
from typing import Union, Tuple


def render_360(
    vertices,
    faces,
    textures_tensor: Union[torch.Tensor, pytorch3d.renderer.TexturesVertex],
    device: torch.device = None,
    fps: int = 10,
    duration: int = 3,
    elev: float = 0,
    azim: float = 0,
    distance: float = 2.732,
    output_path: str = "360.gif",
    resolution: int = 512,
    verbose: bool = False,
) -> Path:
    """
    Render a 360 degree gif of a given mesh with
    specified FPS and duration.

    Returns:
        output_path (Path): Path to the saved gif.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vertices = vertices.to(device)
    faces = faces.to(device)
    textures_tensor = textures_tensor.to(device)

    textures = (
        pytorch3d.renderer.TexturesVertex(
            verts_features=textures_tensor,
        )
        if isinstance(textures_tensor, torch.Tensor)
        else textures_tensor
    )

    # Initialize the renderer
    renderer = get_mesh_renderer(resolution, device=device)

    # Initialize the camera
    R, T = pytorch3d.renderer.cameras.look_at_view_transform(distance, elev, azim)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R,
        T=T,
        fov=60,
        device=device,
    )

    # Initialize the lights
    lights = pytorch3d.renderer.PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=textures,
    )

    # Render the mesh
    images = []
    pbar = tqdm(
        range(0, 360, 360 // (fps * duration)), desc="Rendering", disable=not verbose
    )
    for i in pbar:
        R, T = pytorch3d.renderer.cameras.look_at_view_transform(
            distance, elev, azim + i
        )
        # Update the camera position and Rotation
        cameras.R = R.to(device)
        cameras.T = T.to(device)

        rendered_image = renderer(
            mesh,
            cameras=cameras,
            lights=lights,
        )
        img_ = rendered_image[0, ..., :3].cpu().numpy()
        img_ = (img_ * 255).astype(np.uint8)

        images.append(img_)

    # Save the gif
    imageio.mimsave(output_path, images, duration=1000 / fps, loop=0)

    return output_path
