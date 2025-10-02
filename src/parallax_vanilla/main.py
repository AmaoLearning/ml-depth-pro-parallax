import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from input_model import Mouse
from parallax import Parallax
from client import ParallaxDisplay

def run(args):
    offset_bound = args.offset_bound
    gamma = args.gamma
    output_path = str(args.output_path) + f"/{args.offset_bound}_{args.gamma}/"

    depth = np.load(args.depth_path)["depth"]
    print(f"\nLoad depth data from file: {args.depth_path}")
    print(f"Data type: {type(depth)}, data size: {depth.shape}, data for example: {depth[0][0]}")
    print(f"Extreme values: {np.max(depth):.2f} {np.min(depth):.2f}")

    image = Image.open(args.image_path)
    image = np.array(image)
    print(f"\nLoad imge from {args.image_path}, data size: {depth.shape}, , data for example: {image[0][0]}")

    if image.shape[:2] != depth.shape:
        print(f'\nImage has a shape of {image.shape}, but depth is {depth.shape}')
        return
    
    # image_shape = image.shape[:2]

    # mouse = Mouse(image_shape)

    par = Parallax(image, depth, offset_bound, gamma)
    # par.plot_n_depth_hist()
    display = ParallaxDisplay(par)

    # Path.mkdir(Path(output_path), exist_ok=True)

    # while (input("Enter 'x' to finish: ") != 'x'):
        # dx, dy = mouse.get_move_distance()
        # result = compute_parallax(image, depth, dx, dy, offset_bound, gamma)

        # output_file = output_path + str(args.image_path.stem) + f"_{dx:.2f}_{dy:.2f}.jpg"
        # print(f"Saving parallax {dx:.2f}, {dy:.2f} to: {output_file}")
        # Image.fromarray(result).save(
        #     output_file, format="JPEG", quality=90
        # )
    print("Press Esc to quit.")
    display.run()


def main():
    parser = argparse.ArgumentParser(
        description="Vanilla parallax model from scratch."
    )
    parser.add_argument(
        "-i", 
        "--image-path", 
        type=Path, 
        help="Path to input vanilla image.",
    )

    parser.add_argument(
        "-d",
        "--depth-path",
        type=Path,
        help="Path to input depth data numpy file.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        default="./outputs/",
        help="Path to store output files.",
    )
    parser.add_argument(
        "-b",
        "--offset-bound",
        type=float,
        default=0.2,
        help="The factor bounding parallax, ranging in [0, 0.5]."
    )
    parser.add_argument(
        "-g",
        "--gamma",
        type=float,
        default=0.1,
        help="Exponet controling offset-depth. Tests found that gamma close to 0 is better."
    )

    run(parser.parse_args())



if __name__ == "__main__":
    main()