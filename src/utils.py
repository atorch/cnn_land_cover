import yaml

from sklearn.preprocessing import LabelEncoder

from constants import CDL_MAPPING_FILE


def get_label_encoder_and_mapping():

    with open(CDL_MAPPING_FILE, "r") as infile:

        cdl_mapping = yaml.safe_load(infile)

    label_encoder = LabelEncoder()

    cdl_classes = list(cdl_mapping.keys())
    label_encoder.fit(cdl_classes)

    return label_encoder, cdl_mapping


def get_config(model_config):

    with open(model_config, "r") as infile:

        config = yaml.safe_load(infile)

    assert len(set(config["training_scenes"])) == len(config["training_scenes"])
    assert len(set(config["validation_scenes"])) == len(config["validation_scenes"])
    assert len(set(config["test_scenes"])) == len(config["test_scenes"])

    assert (
        len(set(config["training_scenes"]).intersection(config["validation_scenes"]))
        == 0
    )
    assert (
        len(set(config["test_scenes"]).intersection(config["validation_scenes"])) == 0
    )

    # TODO Also assert that training scenes don't intersect test or validation scenes
    # NAIP scenes can overlap by a few hundred meters

    return config
