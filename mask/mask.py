import os
import rasterio
import numpy as np
from tqdm import tqdm

input_mask = r"D:\Super_Resolution\data\buildings\buildings_delft_lr.tif"
tile_folder = r"C:\Users\mike_\OneDrive\Desktop\MSc Geomatics\Master Thesis\Codebases\SRGAN_CustomDataset\custom_dataset\It1\train_HR4"
output_folder = r"C:\Users\mike_\OneDrive\Desktop\MSc Geomatics\Master Thesis\Codebases\SRGAN_CustomDataset\mask\train_HR2"
os.makedirs(output_folder, exist_ok=True)

tile_size = 256
overlap = int(tile_size * 0.10)  # 10% = 25
window_size = tile_size + 2 * overlap  # 306

tile_list = [f for f in os.listdir(tile_folder) if f.endswith('.tif') and f.startswith('tile_')]

with rasterio.open(input_mask) as mask_src:
    for tile_name in tqdm(tile_list, desc="Processing tiles"):
        tile_path = os.path.join(tile_folder, tile_name)

        with rasterio.open(tile_path) as tile_src:
            bounds = tile_src.bounds
            rgb = tile_src.read()

            row_start, col_start = mask_src.index(bounds.left, bounds.top)
            row_start -= overlap
            col_start -= overlap

            row_start = max(row_start, 0)
            col_start = max(col_start, 0)

            window = rasterio.windows.Window(
                col_off=col_start,
                row_off=row_start,
                width=window_size,
                height=window_size
            )

            mask = mask_src.read(1, window=window)

            center_row = (mask.shape[0] - tile_size) // 2
            center_col = (mask.shape[1] - tile_size) // 2
            mask_cropped = mask[center_row:center_row + tile_size, center_col:center_col + tile_size]

            if mask_cropped.shape != (tile_size, tile_size):
                print(f"Skipping {tile_name}: cropped mask shape {mask_cropped.shape}")
                continue

            combined = np.vstack([rgb, mask_cropped[np.newaxis, :, :]])

            profile = tile_src.profile.copy()
            profile.update({
                'count': 4,
                'dtype': 'uint8',
                'compress': 'DEFLATE'
            })

            out_path = os.path.join(output_folder, tile_name)
            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(combined)
