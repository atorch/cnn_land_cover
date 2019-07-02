NAIP_DIR = "./naip"

BUILDING_ANNOTATION_DIR = "./building_annotations"

CDL_ANNOTATION_DIR = "./cdl_annotations"
# TODO Make sure CDL year matches NAIP year
CDL_FILE = "2017_30m_cdls.img"
CDL_DIR = "./cdl"

COUNTY_DIR = "./county"
COUNTY_FILE = "tl_2017_us_county.shp"

ROAD_ANNOTATION_DIR = "./road_annotations"
ROAD_DIR = "./roads"
ROAD_FORMAT = "tl_2017_{county}_roads.shp"

# Note: buffering is applied after projecting road shapes into NAIP's CRS
ROAD_BUFFER_METERS = 2.0
