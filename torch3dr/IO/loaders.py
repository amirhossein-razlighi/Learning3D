from pathlib import Path
from typing import Tuple, Optional
import torch
from pytorch3d.io import load_obj, load_ply


def load_obj_mesh(
    mesh_path: Path, return_aux: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Loads a mesh from an .obj file.
    Args:
        mesh_path (Path): path to the .obj file
        return_aux (bool): whether to return auxiliary information such as normals and textures
    Returns:
        vertices (Tensor): vertices of the mesh
        faces (Tensor): faces of the mesh
        aux (Tensor): auxiliary information such as normals, textures
    """
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
