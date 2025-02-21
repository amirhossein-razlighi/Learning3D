import imageio
import numpy as np
import pytorch3d
import torch
from PIL import Image, ImageDraw
from tqdm.auto import tqdm
import click

from .rendering_utils import get_device, get_mesh_renderer


def dolly_zoom(
    input_file,
    image_size=256,
    num_frames=10,
    duration=3,
    device=None,
    output_file="output/dolly.gif",
    dolly_in=False,
):
    """
    Create a dolly zoom effect by changing the field of view of the camera.
    Args:
        input_file (str): The input file to load the mesh
        image_size (int, optional): size of output (rendered) image. Defaults to 256.
        num_frames (int, optional): number of frames in the output gif. Defaults to 10.
        duration (int, optional): The duration of the final gif (in seconds). Defaults to 3.
        device (_type_, optional): The device to use. Defaults to None. (Automatically inferred)
        output_file (str, optional): The output file to save the gif. Defaults to "output/dolly.gif".
        dolly_in (bool, optional): Whether to dolly in or out. Defaults to False.
    """
    if device is None:
        device = get_device()

    mesh = pytorch3d.io.load_objs_as_meshes([input_file], device=device)

    renderer = get_mesh_renderer(rendered_image_size=image_size, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0.0, 0.0, -3.0]], device=device)

    fovs = (
        torch.linspace(5, 120, num_frames)
        if dolly_in
        else torch.linspace(120, 5, num_frames)
    )

    renders = []
    for fov in tqdm(fovs):
        distance = 5.0 / (2 * torch.tan(torch.deg2rad(fov) / 2))
        T = [[0.0, 0.0, distance]]
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(fov=fov, T=T, device=device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend[0, ..., :3].cpu().numpy()  # (N, H, W, 3)
        renders.append(rend)
    images = []
    for i, r in enumerate(renders):
        image = Image.fromarray((r * 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)
        draw.text((20, 20), f"fov: {fovs[i]:.2f}", fill=(255, 0, 0))
        images.append(np.array(image))

    imageio.mimsave(output_file, images, duration=duration, loop=0)
