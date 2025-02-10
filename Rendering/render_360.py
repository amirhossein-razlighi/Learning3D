from rendering_utils import *
import pytorch3d
import imageio
import torch
from pathlib import Path
import click


def render_360(
    vertices,
    faces,
    textures_tensor,
    device: torch.device = None,
    fps: int = 10,
    duration: int = 3,
    elev: float = 0,
    azim: float = 0,
    distance: float = 2.732,
    output_path: str = "360.mp4",
    resolution: int = 512,
    verbose: bool = False,
) -> Path:
    """
    Render a 360 degree gif of a given mesh with
    specified FPS and duration.

    Returns:
        path to the output gif file
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vertices = vertices.to(device)
    faces = faces.to(device)
    textures_tensor = textures_tensor.to(device)

    textures = pytorch3d.renderer.TexturesVertex(
        verts_features=textures_tensor,
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
    for i in range(0, 360, 360 // (fps * duration)):
        R, T = pytorch3d.renderer.cameras.look_at_view_transform(
            distance, elev, azim + i
        )
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R,
            T=T,
            fov=60,
            device=device,
        )
        rendered_image = renderer(
            mesh,
            cameras=cameras,
            lights=lights,
        )
        images.append(rendered_image[0, ..., :3].cpu().numpy())

    # Save the images as a gif
    imageio.mimsave(output_path, images, fps=fps)

    return Path(output_path)


@click.command()
@click.option("--mesh_path", type=str, required=True)
@click.option("--output_path", type=str, required=False, default="")
def main(mesh_path, output_path):
    mesh_path = Path(mesh_path)
    output_path = Path(output_path)
    vertices, faces = load_external_mesh(mesh_path)
    vertices = vertices.unsqueeze(0)
    faces = faces.unsqueeze(0)
    textures = torch.ones_like(vertices)
    output_path = output_path if output_path != "" else Path("360.mp4")
    render_360(vertices, faces, textures, output_path=output_path)


if __name__ == "__main__":
    main()
