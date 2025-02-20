import torch
import pytorch3d.structures


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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
