from functools import partial
import glob
import json
import os
import pyproj
import rasterio
from shapely.geometry import Polygon, shape
from shapely.ops import transform

from annotate_naip_scenes import NAIP_DIR


def get_buildings(path):

    with open(path, "r") as infile:

        geojson = json.load(infile)

    buildings = geojson["features"]
    print(f"done loading {len(buildings)} buildings from {path}")

    return buildings


def get_naip_scenes():

    naip_paths = glob.glob(os.path.join(NAIP_DIR, "m_*tif"))
    print(f"found {len(naip_paths)} naip scenes")

    naip_scenes = {}

    for naip_path in naip_paths:

        naip = rasterio.open(naip_path)

        projection_fn = partial(
            pyproj.transform, pyproj.Proj(naip.crs), pyproj.Proj("epsg:4326")
        )

        # TODO Is rasterio bounds convention different from shapely's (xy order switched?)
        minx, miny, maxx, maxy = naip.bounds

        bbox_shape = Polygon([[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy]])

        # Note: projected_bbox_shape.exterior.coords.xy are in reverse order (y then x)  # TODO Why?
        projected_bbox_shape = transform(projection_fn, bbox_shape)

        minlat, minlon, maxlat, maxlon = projected_bbox_shape.bounds

        bbox_shape_lonlat = Polygon(
            [[minlon, minlat], [maxlon, minlat], [maxlon, maxlat], [minlon, maxlat]]
        )

        naip_scenes[naip_path] = {"buildings": [], "bbox_lonlat": bbox_shape_lonlat}

    return naip_scenes


def add_buildings_to_naip_scenes(buildings, naip_scenes):

    for building in buildings:

        building_shape = shape(building["geometry"])

        # Note: a single building can intersect multiple NAIP scenes
        # TODO Only need to loop over naip scenes in the same state!
        for naip_path in naip_scenes.keys():

            bbox_lonlat = naip_scenes[naip_path]["bbox_lonlat"]

            if bbox_lonlat.intersects(building_shape):

                naip_scenes[naip_path]["buildings"].append(building_shape)

    return


def main():

    naip_scenes = get_naip_scenes()

    for state in ["Iowa", "Minnesota"]:

        buildings = get_buildings(f"./buildings/{state}.geojson")

        add_buildings_to_naip_scenes(buildings, naip_scenes)

        print({naip_path: len(value["buildings"]) for naip_path, value in naip_scenes.items()})


if __name__ == "__main__":
    main()
