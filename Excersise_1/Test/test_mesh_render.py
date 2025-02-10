from rendering_utils import *
import torch
import numpy as np
import unittest
import pytorch3d
import matplotlib.pyplot as plt


class TestMeshRender(unittest.TestCase):
    def setUp(self):
        self.device = get_device()
        self.batch_size = 2
        self.num_vertices = 5
        self.num_faces = 3

    def test_mesh_render(self):
        vertices, faces = load_external_mesh(Path("Data/cow.obj"))
        vertices = vertices.unsqueeze(0).repeat(self.batch_size, 1, 1).to(self.device)
        faces = faces.unsqueeze(0).repeat(self.batch_size, 1, 1).to(self.device)
        image_size = 256
        renderer = get_mesh_renderer(rendered_image_size=image_size)
        textures = torch.ones_like(vertices, device=self.device)
        mesh = pytorch3d.structures.Meshes(
            verts=vertices,
            faces=faces,
            textures=pytorch3d.renderer.TexturesVertex(textures),
        )
        lights = pytorch3d.renderer.PointLights(
            location=[[0.0, 0.0, -3.0]],
            device=self.device,
        )
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=torch.eye(3).unsqueeze(0).repeat(self.batch_size, 1, 1),
            T=torch.tensor([[0, 0, 3]]),
            fov=60,
            device=self.device,
        )
        image = renderer(mesh, cameras=cameras, lights=lights)
        self.assertEqual(
            image.shape, (self.batch_size, image_size, image_size, 4)
        )  # RGBA
        plt.axis("off")
        plt.imsave(
            "Test/test_mesh_render.png", image[0].detach().cpu().numpy()[..., :3]
        )
        plt.close()
