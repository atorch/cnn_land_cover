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

# Note: any CDL class absent from CDL_MAPPING_FILE is coded as CDL_CLASS_OTHER
CDL_MAPPING_FILE = "./config/cdl_classes.yml"
CDL_CLASS_OTHER = "other"
CDL_CLASS_BUILDING = "building"
CDL_CLASS_ROAD = "road"

COUNTY_DIR = "./county"
COUNTY_FILE = "tl_2017_us_county.shp"

ROAD_ANNOTATION_DIR = "./road_annotations"
ROAD_DIR = "./roads"
ROAD_FORMAT = "tl_2017_{county}_roads.shp"

# Note: buffering is applied after projecting road shapes into NAIP's CRS
ROAD_BUFFER_METERS = 2.0

HAS_BUILDINGS = "has_buildings"
HAS_ROADS = "has_roads"
IS_MAJORITY_FOREST = "is_majority_forest"
MODAL_LAND_COVER = "modal_land_cover"
PIXELS = "pixels"
