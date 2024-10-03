from argparse import ArgumentParser
from pathlib import Path

import laspy as lp


def tile_las(file: Path, output_folder: Path, tile_size: int):
    # Read the file
    data = lp.read(str(file))
    x_min, x_max = data.header.x_min, data.header.x_max
    y_min, y_max = data.header.y_min, data.header.y_max

    x_range = x_max - x_min
    y_range = y_max - y_min

    x_step = x_range / tile_size
    y_step = y_range / tile_size

    for i in range(tile_size):
        for j in range(tile_size):
            x_min_tile = x_min + i * x_step
            x_max_tile = x_min + (i + 1) * x_step
            y_min_tile = y_min + j * y_step
            y_max_tile = y_min + (j + 1) * y_step

            points = data.points[
                (data.x >= x_min_tile)
                & (data.x <= x_max_tile)
                & (data.y >= y_min_tile)
                & (data.y <= y_max_tile)
            ]
            if len(points) == 0:
                continue

            new_file = lp.create(
                point_format=data.header.point_format, file_version=data.header.version
            )
            new_file.points = points
            new_file.header = data.header
            new_file.header.x_max = x_max_tile
            new_file.header.x_min = x_min_tile
            new_file.header.y_max = y_max_tile
            new_file.header.y_min = y_min_tile
            new_file.header.z_max = points["z"].max()
            new_file.header.z_min = points["z"].min()
            new_file.header.point_records_count = len(points)

            new_file.write(str(output_folder / f"{file.stem}_{i}_{j}.las"))
    return


parser = ArgumentParser()

parser.add_argument(
    "--input", type=Path, required=True, help="Input folder or file", dest="INPUT"
)
parser.add_argument(
    "--tile", type=int, default=3, help="Number of tile per dimension", dest="TILE"
)
parser.add_argument("--out", type=Path, required=True, help="Output folder", dest="OUT")

args = parser.parse_args()

args.OUT.mkdir(parents=True, exist_ok=True)

if args.INPUT.is_dir():
    for file in args.INPUT.iterdir():
        tile_las(file, args.OUT, args.TILE)
else:
    tile_las(args.INPUT, args.OUT, args.TILE)
