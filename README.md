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
python src/annotate_naip_scenes.py
python src/fit_model.py
```

# TODO

* [ ] Download script for the NAIP scenes in [config.yml](config.yml)
* [ ] Env var for year, use it in all download scripts
* [ ] Qix spatial index files for shapefiles
* [ ] More objectives: buildings, semantic segmentation...
* [ ] GPU
* [ ] Test set confusion matrices
* [ ] Tensorboard
* [ ] Tune dropout probability, number of filters, number of blocks

# Datasets

[Census TIGER Shapefiles](https://www.census.gov/geo/maps-data/data/tiger-line.html) (roads and counties)

[Cropland Data Layer](https://www.nass.usda.gov/Research_and_Science/Cropland/Release/) (land cover raster, visualize it [here](https://nassgeodata.gmu.edu/CropScape/))

[National Agriculture Imagery Program](https://www.fsa.usda.gov/programs-and-services/aerial-photography/imagery-programs/naip-imagery/) (four-band aerial imagery, originally downloaded from [USGS Earth Explorer](https://earthexplorer.usgs.gov/?))

The Cropland Data Layer (CDL) and TIGER road shapefiles are used to programmatically generate
labels for NAIP images. See [sample_images](sample_images) for sample input images along
with their labels (one label per objective).

# Models

Mini [VGG-ish model](cnn.py) with three objectives:

```
Classification report for is_majority_forest:
              precision    recall  f1-score   support

           0       0.96      0.98      0.97       402
           1       0.90      0.84      0.87        98

    accuracy                           0.95       500
   macro avg       0.93      0.91      0.92       500
weighted avg       0.95      0.95      0.95       500

Classification report for has_roads:
              precision    recall  f1-score   support

           0       0.94      0.95      0.95       338
           1       0.89      0.88      0.89       162

    accuracy                           0.93       500
   macro avg       0.92      0.91      0.92       500
weighted avg       0.93      0.93      0.93       500

Classification report for modal_land_cover:
              precision    recall  f1-score   support

    corn_soy       0.78      0.91      0.84       216
   developed       0.83      0.74      0.78        34
      forest       0.83      0.88      0.85       103
       other       0.00      0.00      0.00         0
     pasture       0.12      0.10      0.11        42
       water       0.94      0.94      0.94        18
    wetlands       0.24      0.64      0.35        11

   micro avg       0.68      0.80      0.74       424
   macro avg       0.54      0.60      0.55       424
weighted avg       0.72      0.80      0.76       424
 samples avg       0.68      0.68      0.68       424
```