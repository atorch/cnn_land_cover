from functools import partial
import glob
from multiprocessing import Pool
import os
import subprocess

import fiona
import numpy as np
import pyproj
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.geometry import Polygon, shape
from shapely.ops import transform

from constants import (
    BUILDING_ANNOTATION_DIR,
    COUNTY_DIR,
    CDL_ANNOTATION_DIR,
    CDL_DIR,
    CDL_FILE,
    COUNTY_FILE,
    NAIP_DIR,
    ROAD_ANNOTATION_DIR,
)


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


def save_cdl_annotation_for_naip_raster(cdl, naip_file, naip):

    # Note: the CDL annotation for a given naip_file has the same file name,
    # but is saved to a different directory
    output_path = os.path.join(CDL_ANNOTATION_DIR, naip_file)

    if os.path.exists(output_path):
        print(f"{output_path} already exists, skipping")
        return

    y_naip, x_naip = get_y_x_at_pixel_centers(naip)

    proj_naip = pyproj.Proj(naip.crs)
    proj_cdl = pyproj.Proj(cdl.crs)

    x_cdl, y_cdl = pyproj.transform(proj_naip, proj_cdl, x_naip, y_naip)

    cdl_values = get_raster_values(cdl, x_cdl, y_cdl)

    cdl_values = cdl_values.reshape((naip.meta["height"], naip.meta["width"]))

    profile = naip.profile.copy()

    # Note: the output has the same width, height, and transform as the NAIP raster,
    # but contains a single band of CDL codes (whereas the NAIP raster contains 4 bands)
    profile["dtype"] = cdl.profile["dtype"]
    profile["count"] = 1

    print(f"writing {output_path}")

    if not os.path.exists(CDL_ANNOTATION_DIR):
        os.makedirs(CDL_ANNOTATION_DIR)

    with rasterio.open(output_path, "w", **profile) as output:
        output.write(cdl_values.astype(profile["dtype"]), 1)


def save_building_annotation_for_naip_raster(naip_file, naip):

    output_path = os.path.join(BUILDING_ANNOTATION_DIR, naip_file)

    if os.path.exists(output_path):
        print(f"{output_path} already exists, skipping")
        return

    building_file = naip_file.replace(".tif", ".shp")
    building_path = os.path.join(BUILDING_ANNOTATION_DIR, building_file)
    building_shp = fiona.open(building_path)

    building_geometries = []

    # Note: we project from the building shapefile's CRS to the NAIP raster's CRS
    projection_fn = partial(
        pyproj.transform, pyproj.Proj(building_shp.crs), pyproj.Proj(naip.crs)
    )

    for building in building_shp:

        building_geometry = shape(building["geometry"])

        building_geometry_transformed = transform(projection_fn, building_geometry)

        building_geometries.append(building_geometry_transformed)

    if building_geometries:
        building_values = rasterize(
            building_geometries,
            out_shape=(naip.meta["height"], naip.meta["width"]),
            transform=naip.transform,
            all_touched=True,
            dtype="uint8",
        )
    else:
        building_values = np.zeros(
            (naip.meta["height"], naip.meta["width"]), dtype="uint8"
        )

    profile = naip.profile.copy()

    # Note: the output has the same width, height, and transform as the NAIP raster,
    # but contains a single band of building indicators (whereas the NAIP raster contains 4 bands)
    profile["dtype"] = "uint8"
    profile["count"] = 1

    print(f"writing {output_path}")
    with rasterio.open(output_path, "w", **profile) as output:
        output.write(building_values.astype(profile["dtype"]), 1)


def save_road_annotation_for_naip_raster(counties, naip_file, naip):

    # Note: the road annotation for a given naip_file has the same file name,
    # but is saved to a different directory
    output_path = os.path.join(ROAD_ANNOTATION_DIR, naip_file)

    if os.path.exists(output_path):
        print(f"{output_path} already exists, skipping")
        return

    road_geometries = []

    for county in counties:

        road_file = os.path.join(ROAD_DIR, ROAD_FORMAT.format(county=county))

        if not os.path.exists(road_file):
            print(f"{road_file} does not exist, downloading it now")
            subprocess.call(["./scripts/download_road_shapefile.sh", county])

        road_shp = fiona.open(road_file)

        # Note: we project from the road shapefile's CRS to the NAIP raster's CRS
        projection_fn = partial(
            pyproj.transform, pyproj.Proj(road_shp.crs), pyproj.Proj(naip.crs)
        )

        for road in road_shp:

            # TODO Census road shapefiles include road types / codes
            # Do they distinguish paved versus unpaved roads?  If so, treat them differently?
            road_geometry = shape(road["geometry"])

            road_geometry_transformed = transform(projection_fn, road_geometry)

            road_geometries.append(road_geometry_transformed.buffer(ROAD_BUFFER_METERS))

    road_values = rasterize(
        road_geometries,
        out_shape=(naip.meta["height"], naip.meta["width"]),
        transform=naip.transform,
        all_touched=True,
        dtype="uint8",
    )

    # TODO Copied from CDL code, put in a function
    profile = naip.profile.copy()

    # Note: the output has the same width, height, and transform as the NAIP raster,
    # but contains a single band of road indicators (whereas the NAIP raster contains 4 bands)
    profile["dtype"] = "uint8"
    profile["count"] = 1

    print(f"writing {output_path}")

    if not os.path.exists(ROAD_ANNOTATION_DIR):
        os.makedirs(ROAD_ANNOTATION_DIR)

    with rasterio.open(output_path, "w", **profile) as output:
        output.write(road_values.astype(profile["dtype"]), 1)


def get_counties(raster):

    # TODO Qix for county files
    county_shp = fiona.open(os.path.join(COUNTY_DIR, COUNTY_FILE))

    width, height = (raster.meta["width"], raster.meta["height"])

    # Note: this is the (x, y) at the top-left of the raster
    x, y = (raster.transform.c, raster.transform.f)

    resolution_x, resolution_y = (raster.transform.a, raster.transform.e)

    raster_bbox = Polygon(
        [
            [x, y],
            [x + resolution_x * width, y],
            [x + resolution_x * width, y + resolution_y * height],
            [x, y + resolution_y * height],
        ]
    )

    # Note: we project from the raster's CRS to the county shapefile's CRS
    projection_fn = partial(
        pyproj.transform, pyproj.Proj(raster.crs), pyproj.Proj(county_shp.crs)
    )

    raster_bbox_reprojected = transform(projection_fn, raster_bbox)

    candidate_counties = county_shp.items(bbox=raster_bbox_reprojected.bounds)

    counties = []

    for _id, candidate_county in candidate_counties:

        if shape(candidate_county["geometry"]).intersects(raster_bbox_reprojected):

            counties.append(candidate_county["properties"]["GEOID"])

    return counties


def save_naip_annotations(naip_path):

    cdl_path = os.path.join(CDL_DIR, CDL_FILE)
    cdl = rasterio.open(cdl_path)

    print(f"annotating {naip_path}")

    naip = rasterio.open(naip_path)

    naip_file = os.path.split(naip_path)[-1]

    counties = get_counties(naip)
    print(f"counties: {', '.join(counties)}")

    save_building_annotation_for_naip_raster(naip_file, naip)
    save_cdl_annotation_for_naip_raster(cdl, naip_file, naip)
    save_road_annotation_for_naip_raster(counties, naip_file, naip)


def main():

    naip_paths = glob.glob(os.path.join(NAIP_DIR, "m_*tif"))
    print(f"found {len(naip_paths)} naip scenes")

    with Pool(10) as pool:

        pool.map(save_naip_annotations, naip_paths)


if __name__ == "__main__":
    main()
