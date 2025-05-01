import rasterio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def read_tile_rgb_and_mask(path):
    with rasterio.open(path) as src:
        data = src.read()  # shape: (bands, height, width)
        rgb = np.transpose(data[:3], (1, 2, 0))  # RGB for display
        mask = data[3] if data.shape[0] > 3 else None  # 4th band as mask
    return rgb, mask

def plot_tiles(hr_rgb, hr_mask, lr_rgb, lr_mask):
    fig, axs = plt.subplots(1, 4, figsize=(18, 5))

    axs[0].imshow(hr_rgb)
    axs[0].set_title("HR Tile (RGB)")
    axs[0].axis("off")

    axs[1].imshow(hr_mask, cmap="gray")
    axs[1].set_title("HR Mask (Band 4)")
    axs[1].axis("off")

    axs[2].imshow(lr_rgb)
    axs[2].set_title("LR Tile (RGB)")
    axs[2].axis("off")

    axs[3].imshow(lr_mask, cmap="gray")
    axs[3].set_title("LR Mask (Band 4)")
    axs[3].axis("off")

    plt.tight_layout()
    plt.show()

def main():
    hr_path = r"C:\Users\mike_\OneDrive\Desktop\MSc Geomatics\Master Thesis\Codebases\SRGAN_CustomDataset\mask\train_HR2\tile_1150_10120.tif"
    lr_path = r"C:\Users\mike_\OneDrive\Desktop\MSc Geomatics\Master Thesis\Codebases\SRGAN_CustomDataset\mask\train_LR2\tile_1150_10120_down.tif"

    hr_rgb, hr_mask = read_tile_rgb_and_mask(hr_path)
    lr_rgb, lr_mask = read_tile_rgb_and_mask(lr_path)

    if hr_mask is None or lr_mask is None:
        print("Error: One or more tiles don't have a 4th band for mask.")
        return

    plot_tiles(hr_rgb, hr_mask, lr_rgb, lr_mask)

if __name__ == "__main__":
    main()
