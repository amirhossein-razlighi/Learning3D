from pathlib import Path
from typing import Tuple, Optional, Union

import torch
import pytorch3d
from pytorch3d.io import load_obj, load_ply, load_objs_as_meshes

from torch3dr.utils import get_device


def load_obj_mesh(
    mesh_path: Path,
    return_aux: bool = False,
    return_type: str = "pt",
    device: str = None,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
    pytorch3d.structures.meshes.Meshes,
]:
    """
    Loads a mesh from an .obj file.
    Args:
        mesh_path (Path): path to the .obj file
        return_aux (bool): whether to return auxiliary information such as normals and textures
        return_type (str): return type of the vertices and faces. Can be either "pt" or "mesh". "pt"
            return pytorch tensors as tuple, and "mesh" returns a Pytorch3D Meshes object.
    Returns:
        (if return_type == "pt")
        vertices (Tensor): vertices of the mesh
        faces (Tensor): faces of the mesh
        aux (Tensor): auxiliary information such as normals, textures

        (if return_type == "mesh")
        mesh (Meshes): Pytorch3D Meshes object
    """
    if return_type not in ["pt", "mesh"]:
        raise ValueError(f"Unsupported return type: {return_type}")
    if device is None:
        device = get_device()
    if return_type == "mesh":
        mesh = load_objs_as_meshes([mesh_path], device=device)
        return mesh

    verts, faces, aux = load_obj(mesh_path)
    return verts, faces.verts_idx, aux if return_aux else None


def load_ply_mesh(mesh_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Loads a mesh from a .ply file.
    Args:
        mesh_path (Path): path to the .ply file
    Returns:
        vertices (Tensor): vertices of the mesh
        faces (Tensor): faces of the mesh
    """

    verts, faces = load_ply(mesh_path)
    return verts, faces.verts_idx


def load_external_mesh(
    mesh_path: Path, return_aux: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Loads an external mesh from a file. Automatically inferes the files extenstion
    and uses the appropriate loader.
    supported formats: [.obj, .ply]
    Args:
        mesh_path (Path): path to the mesh file
    Returns:
        vertices (Tensor): vertices of the mesh
        faces (Tensor): faces of the
    """

    aux = None
    if mesh_path.suffix == ".obj":
        vertices, faces, aux = load_obj_mesh(mesh_path, return_aux)
    elif mesh_path.suffix == ".ply":
        vertices, faces = load_ply_mesh(mesh_path)
    else:
        raise ValueError(f"Unsupported file format: {mesh_path.suffix}")

    return vertices, faces, aux
