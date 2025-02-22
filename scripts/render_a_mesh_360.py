import click
from pathlib import Path
import torch
from torch3dr.io import load_external_mesh
from torch3dr.rendering import render_360


@click.command()
@click.option("--mesh_path", type=str, required=True)
@click.option("--output_path", type=str, required=False, default="")
@click.option("--verbose", is_flag=True)
def main(mesh_path: str, output_path: str, verbose: bool):
    mesh_path = Path(mesh_path)
    vertices, faces = load_external_mesh(mesh_path)
    vertices = vertices.unsqueeze(0)
    faces = faces.unsqueeze(0)
    textures = torch.ones_like(vertices)
    output_path = output_path if output_path != "" else "obj_360_rendered.gif"
    render_360(vertices, faces, textures, output_path=output_path, verbose=verbose)


if __name__ == "__main__":
    main()
