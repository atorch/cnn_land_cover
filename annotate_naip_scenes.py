import glob
import os

import numpy as np
import pyproj
import rasterio
from rasterio.windows import Window


CDL_DIR = "./cdl"
COUNTY_DIR = "./county"
NAIP_DIR = "./naip"

# TODO Make sure CDL year matches NAIP year
CDL_FILE = "2017_30m_cdls.img"
COUNTY_FILE = "tl_2018_us_county.shp"

CDL_ANNOTATION_PREFIX = "cdl_for_"

# Note: any CDL class absent from CDL_MAPPING_FILE is coded as CDL_CLASS_OTHER
CDL_MAPPING_FILE = "cdl_classes.yml"
CDL_CLASS_OTHER = "other"


def get_y_x_at_pixel_centers(raster):
    yx_mesh = np.meshgrid(
        range(raster.meta["height"]), range(raster.meta["width"]), indexing="ij"
    )

    y_flat = yx_mesh[0].flatten()
    x_flat = yx_mesh[1].flatten()

    y_raster_proj = (
        raster.meta["transform"].f + (y_flat + 0.5) * raster.meta["transform"].e
    )
    x_raster_proj = (
        raster.meta["transform"].c + (x_flat + 0.5) * raster.meta["transform"].a
    )

    return y_raster_proj, x_raster_proj


def get_raster_values(raster, x_raster_proj, y_raster_proj):

    # raster_values = raster.sample(zip(x_raster_proj.tolist(), y_raster_proj.tolist()))
    # raster_values = [x[0] for x in raster_values]

    min_x = min(x_raster_proj)
    max_x = max(x_raster_proj)

    min_y = min(y_raster_proj)
    max_y = max(y_raster_proj)

    y_offset = (max_y - raster.meta["transform"].f) / raster.meta["transform"].e
    y_offset = np.floor(y_offset).astype(int)
    assert y_offset > 0

    height = (min_y - raster.meta["transform"].f) / raster.meta[
        "transform"
    ].e - y_offset
    height = np.ceil(height).astype(int)
    assert height > 0

    x_offset = (min_x - raster.meta["transform"].c) / raster.meta["transform"].a
    x_offset = np.floor(x_offset).astype(int)
    assert x_offset > 0

    width = (max_x - raster.meta["transform"].c) / raster.meta["transform"].a - x_offset
    width = np.ceil(width).astype(int)
    assert width > 0

    window = Window(x_offset, y_offset, width, height)

    # Note: shape is (band, y, x), i.e. (band, height, width)
    raster_window_values = raster.read(window=window)

    x_window_index = (x_raster_proj - raster.meta["transform"].c) / raster.meta[
        "transform"
    ].a - x_offset
    x_window_index = x_window_index.astype(int)

    y_window_index = (y_raster_proj - raster.meta["transform"].f) / raster.meta[
        "transform"
    ].e - y_offset
    y_window_index = y_window_index.astype(int)

    return raster_window_values[0, y_window_index, x_window_index]


def save_cdl_values_for_naip_raster(x_cdl, y_cdl, cdl, naip_file, naip):

    cdl_values = get_raster_values(cdl, x_cdl, y_cdl)

    cdl_values = cdl_values.reshape((naip.meta["height"], naip.meta["width"]))

    output_file = CDL_ANNOTATION_PREFIX + naip_file

    profile = naip.profile.copy()

    # Note: the output has the same width, height, and transform as the NAIP raster,
    # but contains a single band of CDL codes (whereas the NAIP raster contains 4 bands)
    profile["dtype"] = cdl.profile["dtype"]
    profile["count"] = 1

    output_path = os.path.join(NAIP_DIR, output_file)

    print(f"writing {output_file}")

    with rasterio.open(output_path, "w", **profile) as output:
        output.write(cdl_values.astype(profile["dtype"]), 1)


def get_counties(raster):
    # TODO Finish this function
    county_shp = fiona.open(os.path.join(COUNTY_DIR, COUNTY_FILE))

    raster_poly = Polygon()

    counties = []

    return counties


def save_naip_annotations(naip_paths):

    cdl_path = os.path.join(CDL_DIR, CDL_FILE)
    cdl = rasterio.open(cdl_path)
    proj_cdl = pyproj.Proj(cdl.crs)

    for naip_path in naip_paths:

        print(f"processing {naip_path}")

        naip = rasterio.open(naip_path)
        proj_naip = pyproj.Proj(naip.crs)

        # counties = get_counties(naip)

        y_naip, x_naip = get_y_x_at_pixel_centers(naip)

        x_cdl, y_cdl = pyproj.transform(proj_naip, proj_cdl, x_naip, y_naip)

        naip_file = os.path.split(naip_path)[-1]

        save_cdl_values_for_naip_raster(x_cdl, y_cdl, cdl, naip_file, naip)


def get_cdl_annotation_path_from_naip_path(naip_path):
    head, tail = os.path.split(naip_path)

    return os.path.join(head, CDL_ANNOTATION_PREFIX + tail)


def main():

    naip_paths = glob.glob(os.path.join(NAIP_DIR, "m_*tif"))
    print(f"found {len(naip_paths)} naip scenes")

    save_naip_annotations(naip_paths)


if __name__ == "__main__":
    main()
