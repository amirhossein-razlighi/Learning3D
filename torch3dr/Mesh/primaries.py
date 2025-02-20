import torch
import pytorch3d.structures
from torch3dr.Rendering import get_device


def create_tetrahedron(device=None):
    """
    Create a tetrahedron mesh.

    The tetrahedron is created by placing 4 vertices:
    - One vertex at the origin (0,0,0)
    - Three vertices forming an equilateral triangle in a plane
    - One vertex above the center of the triangle

    Returns:
        pytorch3d.structures.Meshes: Tetrahedron mesh
    """
    if device is None:
        device = get_device()

    # Calculate vertices
    # We'll use a regular tetrahedron where all faces are equilateral triangles
    a = 2.0  # Edge length

    # Calculate height of the tetrahedron
    h = a * (2 / 3) ** 0.5

    # Calculate radius of circumscribed circle of the base triangle
    r = a / 3**0.5

    # Define vertices
    vertices = torch.tensor(
        [
            [
                r * torch.cos(torch.tensor(0.0)),
                r * torch.sin(torch.tensor(0.0)),
                0,
            ],  # Vertex 0
            [
                r * torch.cos(torch.tensor(2 * torch.pi / 3)),
                r * torch.sin(torch.tensor(2 * torch.pi / 3)),
                0,
            ],  # Vertex 1
            [
                r * torch.cos(torch.tensor(4 * torch.pi / 3)),
                r * torch.sin(torch.tensor(4 * torch.pi / 3)),
                0,
            ],  # Vertex 2
            [0, 0, h],  # Top vertex
        ],
        device=device,
    )

    # Define faces (counter-clockwise order for correct normal orientation)
    faces = torch.tensor(
        [
            [0, 1, 2],  # Bottom face
            [0, 2, 3],  # Side face 1
            [1, 3, 2],  # Side face 2
            [0, 3, 1],  # Side face 3
        ],
        device=device,
    )

    # Create mesh (add batch dimension)
    vertices = vertices.unsqueeze(0)  # (1, 4, 3)
    faces = faces.unsqueeze(0)  # (1, 4, 3)

    # Create mesh with default white color
    textures = torch.zeros_like(vertices)  # (1, 4, 3)

    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )

    return mesh


def create_cube(size=1.0, device=None):
    """
    Create a cube mesh centered at origin.

    Args:
        size (float): Length of cube edges. Defaults to 1.0.
        device (torch.device): Device to place the mesh on.
            Defaults to None (uses get_device()).

    Returns:
        pytorch3d.structures.Meshes: Cube mesh
    """
    if device is None:
        device = get_device()

    # Define vertices (8 corners of the cube)
    vertices = torch.tensor(
        [
            [-1, -1, -1],  # 0: left  bottom back
            [1, -1, -1],  # 1: right bottom back
            [1, 1, -1],  # 2: right top    back
            [-1, 1, -1],  # 3: left  top    back
            [-1, -1, 1],  # 4: left  bottom front
            [1, -1, 1],  # 5: right bottom front
            [1, 1, 1],  # 6: right top    front
            [-1, 1, 1],  # 7: left  top    front
        ],
        device=device,
    ) * (size / 2)

    # Define faces (12 triangles forming 6 square faces)
    faces = torch.tensor(
        [
            [0, 1, 2],  # back face
            [0, 2, 3],
            [4, 6, 5],  # front face
            [4, 7, 6],
            [0, 4, 5],  # bottom face
            [0, 5, 1],
            [2, 6, 7],  # top face
            [2, 7, 3],
            [0, 3, 7],  # left face
            [0, 7, 4],
            [1, 5, 6],  # right face
            [1, 6, 2],
        ],
        device=device,
    )

    # Add batch dimension
    vertices = vertices.unsqueeze(0)  # (1, 8, 3)
    faces = faces.unsqueeze(0)  # (1, 12, 3)

    # Create white textures
    textures = torch.zeros_like(vertices)  # (1, 8, 3)

    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )

    return mesh
