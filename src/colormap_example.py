import numpy as np
import rasterio


def main(naip_path="./naip/m_2708029_ne_17_1_20171211.tif", outpath="./colormap_example.tif"):

    naip = rasterio.open(naip_path)

    X = naip.read()

    # Note: X.shape is (4, 7554, 6802)
    new_X = np.zeros(X.shape[1:], dtype="uint8")

    new_X[:200, :2000] = 1
    new_X[200:1000, :2000] = 2

    profile = naip.profile

    profile["dtype"] = new_X.dtype
    profile["count"] = 1

    colormap = {
        0: (50, 50, 255),  # RGB dark blue, complement is yellow
        1: (255, 255, 0),  # RGB yellow, comeplemnt is blue
        2: (150, 0, 100),  # RGB purple-ish, complement is green
    }

    with rasterio.open(outpath, "w", **profile) as outfile:

        outfile.write(new_X, 1)
        outfile.write_colormap(1, colormap)

if __name__ == "__main__":
    main()
