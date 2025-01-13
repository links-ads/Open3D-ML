from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def tile_ply(file: Path, output_folder: Path, tile_size: int):

    # Leggi il file PLY
    ply_data = PlyData.read(str(file))
    pc_vertex_data = ply_data["vertex"]

    # Estrai le coordinate dei punti
    points = np.column_stack(
        (pc_vertex_data["x"], pc_vertex_data["y"], pc_vertex_data["z"])
    )

    # Calcola i bounds
    x_min, y_min, z_min = points.min(axis=0)
    x_max, y_max, z_max = points.max(axis=0)

    x_range = x_max - x_min
    y_range = y_max - y_min

    x_step = x_range / tile_size
    y_step = y_range / tile_size

    for i in range(tile_size):
        for j in range(tile_size):
            tile_id = i * tile_size + j

            x_min_tile = x_min + i * x_step
            x_max_tile = x_min + (i + 1) * x_step
            y_min_tile = y_min + j * y_step
            y_max_tile = y_min + (j + 1) * y_step

            # Filtra i punti della tile corrente
            mask = (
                (points[:, 0] >= x_min_tile)
                & (points[:, 0] <= x_max_tile)
                & (points[:, 1] >= y_min_tile)
                & (points[:, 1] <= y_max_tile)
            )

            tile_points = points[mask]

            if len(tile_points) == 0:
                continue

            # Prepara i dati per il nuovo file PLY
            tile_vertex_data = []
            for k in pc_vertex_data.dtype().names:
                tile_vertex_data.append(pc_vertex_data[k][mask])
            vertex = np.stack(tile_vertex_data, axis=1)

            vertex = np.array(
                [tuple(r) for r in vertex.tolist()], dtype=pc_vertex_data.dtype()
            )
            el = PlyElement.describe(vertex, "vertex")

            # Salva la tile come file PLY
            output_file = output_folder / f"{file.stem}_{tile_id}.ply"
            PlyData([el], text=True).write(str(output_file))

            print(f"Salvata tile {tile_id} con {len(tile_points)} punti")

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
        tile_ply(file, args.OUT, args.TILE)
else:
    tile_ply(args.INPUT, args.OUT, args.TILE)
