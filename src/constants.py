# Note: this is the image shape used when training
# The image shape during prediction can differ since the model is fully convolutional
IMAGE_SHAPE = (256, 256, 4)

MODEL_CONFIG = "./config/model_config.yml"

NAIP_DIR = "./naip"

BUILDING_ANNOTATION_DIR = "./building_annotations"

CDL_ANNOTATION_DIR = "./cdl_annotations"
# TODO Make sure CDL year matches NAIP year
CDL_FILE = "2017_30m_cdls.img"
CDL_DIR = "./cdl"

# TODO Print which cdl codes are most common within other
CDL_CLASSES_TO_MASK = ["mask", "other"]

# Note: any CDL class absent from CDL_MAPPING_FILE is coded as other
CDL_MAPPING_FILE = "./config/cdl_classes.yml"

COUNTY_DIR = "./county"
COUNTY_FILE = "tl_2017_us_county.shp"

ROAD_ANNOTATION_DIR = "./road_annotations"
ROAD_DIR = "./roads"
ROAD_FORMAT = "tl_2017_{county}_roads.shp"

# Note: buffering is applied after projecting road shapes into NAIP's CRS
# For road type definitions, see https://www2.census.gov/geo/pdfs/maps-data/data/tiger/tgrshp2009/TGRSHP09AF.pdf
ROAD_BUFFER_METERS_DEFAULT = 3.0
ROAD_BUFFER_METERS = {
    "S1100": 5.0,
    "S1200": 5.0,
}

HAS_BUILDINGS = "has_buildings"
HAS_ROADS = "has_roads"
IS_MAJORITY_FOREST = "is_majority_forest"
MODAL_LAND_COVER = "modal_land_cover"
PIXELS = "pixels"
