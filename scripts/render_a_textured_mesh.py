from torch3dr.rendering import render_textured_obj
import matplotlib.pyplot as plt
import click
from scipy.spatial.transform.rotation import Rotation as R
import torch


@click.command()
@click.option("--obj-path", type=str, help="Path to the .obj file")
@click.option("--img-size", type=int, default=256, help="Size of the output image")
def main(obj_path, img_size):
    relative_r = R.from_euler("xyz", [0, 90, 0], degrees=True).as_matrix()
    relative_r = torch.tensor(relative_r, dtype=torch.float32)
    transform_t = torch.tensor([-3, 0, 2.0], dtype=torch.float32).unsqueeze(0)

    img = render_textured_obj(
        obj_path, img_size, R_Relative=relative_r, T_Relative=transform_t
    )
    plt.imshow(img)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
