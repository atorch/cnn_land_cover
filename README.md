# CNN for Land Cover

```bash
./scripts/download_cdl.sh
./scripts/download_county_shapefile.sh
```

```bash
export DOCKER_TAG=cnn_land_cover_docker
sudo docker build ~/cnn_land_cover --tag=$DOCKER_TAG
sudo docker run -it -v ~/cnn_land_cover:/home/cnn_land_cover $DOCKER_TAG bash
cd /home/cnn_land_cover
python annotate_naip_scenes.py
python fit_model.py
```

# TODO

* [ ] Download script for the NAIP scenes in [config.yml](config.yml)
* [ ] Env var for year, use it in all download scripts
* [ ] Qix spatial index files for shapefiles
* [ ] More objectives: buildings, semantic segmentation...

# Datasets

[Census TIGER Shapefiles](https://www.census.gov/geo/maps-data/data/tiger-line.html) (roads and counties)

[Cropland Data Layer](https://www.nass.usda.gov/Research_and_Science/Cropland/Release/) (land cover raster, visualize it [here](https://nassgeodata.gmu.edu/CropScape/))

[National Agriculture Imagery Program](https://www.fsa.usda.gov/programs-and-services/aerial-photography/imagery-programs/naip-imagery/) (four-band aerial imagery, originally downloaded from [USGS Earth Explorer](https://earthexplorer.usgs.gov/?))

# Models

Mini [VGG-ish model](cnn.py) with two objectives:

```
Classification report for is_majority_forest:
              precision    recall  f1-score   support

           0       0.91      0.93      0.92       321
           1       0.87      0.83      0.85       179

    accuracy                           0.89       500
   macro avg       0.89      0.88      0.88       500
weighted avg       0.89      0.89      0.89       500

Classification report for has_roads:
              precision    recall  f1-score   support

           0       0.96      0.96      0.96       423
           1       0.80      0.79      0.80        77

    accuracy                           0.94       500
   macro avg       0.88      0.88      0.88       500
weighted avg       0.94      0.94      0.94       500
```