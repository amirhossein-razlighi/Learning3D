import torch
import pytorch3d
from pytorch3d.renderer import TexturesVertex


def get_texure_vertex_from_tensor(vertices_tensor_texture: torch.tensor):
    """
    Convert a tensor of vertices to a TexturesVertex object
    Args:
        vertices_tensor_texture: tensor of vertices
    Returns:
        TexturesVertex object
    """
    return TexturesVertex(verts_features=vertices_tensor_texture)


def color_interpolate_texture(
    mesh: pytorch3d.structures.Meshes, color1: torch.tensor, color2: torch.tensor
):
    """
    Interpolate between two colors
    Args:
        mesh: mesh object
        color1: first color
        color2: second color
    Returns:
        TexturesVertex object
    """
    if color1.shape[0] != 3 or color2.shape[0] != 3:
        raise ValueError("Color tensors should be RGB color tensor (R, G, B)")

    # Linspace alpha between 0 and 1 based on z coordinate of the mesh vertices
    alpha = (mesh.verts_packed()[:, 2] - mesh.verts_packed()[:, 2].min()) / (
        mesh.verts_packed()[:, 2].max() - mesh.verts_packed()[:, 2].min()
    )
    alpha = alpha.unsqueeze(1).repeat(1, 3)

    color = alpha * color1 + (1 - alpha) * color2
    return TexturesVertex(verts_features=color.unsqueeze(0))
