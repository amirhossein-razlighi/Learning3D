import click
from torch3dr.Rendering import dolly_zoom


@click.command()
@click.option("--image_size", default=256, help="size of output (rendered) image")
@click.option("--num_frames", default=10, help="number of frames in the output gif")
@click.option(
    "--duration", default=3, help="The duration of the final gif (in seconds)"
)
@click.option("--device", default=None, help="The device to use")
@click.option(
    "--output_file", default="output/dolly.gif", help="The output file to save the gif"
)
@click.option("--input_file", required=True, help="The input file to load the mesh")
@click.option("--dolly_in", is_flag=True, help="Whether to dolly in or out")
def main(
    image_size,
    num_frames,
    duration,
    device,
    output_file,
    input_file,
    dolly_in,
):
    dolly_zoom(
        input_file, image_size, num_frames, duration, device, output_file, dolly_in
    )


if __name__ == "__main__":
    main()
