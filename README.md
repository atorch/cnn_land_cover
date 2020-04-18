# CNN for Land Cover

This repo trains a model that predicts [land cover](https://en.wikipedia.org/wiki/Land_cover)
using aerial imagery. This is a [semantic segmentation](https://www.youtube.com/watch?v=nDPWywWRIRo) task:
the inputs to the model look like [this](screenshots/test_set_prediction_screenshot_naip.png),
and the predictions look like [this](screenshots/test_set_prediction_screenshot_opaque.png).
See [here](screenshots/test_set_prediction_screenshot_partially_transparent.png)
for an image showing both the input and the predictions.
These screenshots were taken from a scene in the model's [test set](config/model_config.yml#L36).
Each color in the predictions corresponds to a land cover class:
forests are green, roads are dark grey, and water is blue.

# Running the code

```bash
./scripts/download_buildings.sh
./scripts/download_cdl.sh
./scripts/download_county_shapefile.sh
```

```bash
sudo docker build ~/cnn_land_cover --tag=cnn_land_cover_docker
sudo docker run -it -v ~/cnn_land_cover:/home/cnn_land_cover cnn_land_cover_docker bash
cd /home/cnn_land_cover
python src/save_building_shapefiles.py
python src/annotate_naip_scenes.py
python src/fit_model.py
```

# TODO

* [ ] Download script for the NAIP scenes in [model_config.yml](config/model_config.yml)
* [ ] Env var for year, use it in all download scripts
* [ ] Qix spatial index files for shapefiles
* [ ] GPU
* [ ] Tensorboard
* [ ] Tune dropout probability, number of filters, number of blocks
* [ ] Visualizations, including gradients

# Datasets

[Census TIGER Shapefiles](https://www.census.gov/geo/maps-data/data/tiger-line.html) (roads and counties)

[Cropland Data Layer](https://www.nass.usda.gov/Research_and_Science/Cropland/Release/)
(land cover raster, visualize it [here](https://nassgeodata.gmu.edu/CropScape/))

[Microsoft building footprints](https://github.com/microsoft/USBuildingFootprints)

[National Agriculture Imagery Program](https://www.fsa.usda.gov/programs-and-services/aerial-photography/imagery-programs/naip-imagery/)
(NAIP four-band aerial imagery, originally downloaded from [USGS Earth Explorer](https://earthexplorer.usgs.gov/?))

The Cropland Data Layer (CDL) and TIGER road shapefiles are used to programmatically generate
labels for NAIP images. See [sample_images](sample_images) for sample input images along
with their labels (one label per objective).

# Models

[Fully convolutional neural network](src/cnn.py) with four objectives:

```
Classification report for has_buildings:
              precision    recall  f1-score   support

           0       0.82      0.97      0.89       319
           1       0.96      0.76      0.85       281

    accuracy                           0.87       600
   macro avg       0.89      0.86      0.87       600
weighted avg       0.89      0.87      0.87       600

Classification report for is_majority_forest:
              precision    recall  f1-score   support

           0       0.97      0.97      0.97       526
           1       0.79      0.77      0.78        74

    accuracy                           0.95       600
   macro avg       0.88      0.87      0.88       600
weighted avg       0.95      0.95      0.95       600

Classification report for has_roads:
              precision    recall  f1-score   support

           0       0.94      0.90      0.92       273
           1       0.92      0.95      0.94       327

    accuracy                           0.93       600
   macro avg       0.93      0.93      0.93       600
weighted avg       0.93      0.93      0.93       600

Classification report for modal_land_cover:
              precision    recall  f1-score   support

    building       0.00      0.00      0.00         0
    corn_soy       0.89      0.81      0.85       205
   developed       0.87      0.84      0.86       135
      forest       0.84      0.83      0.84        89
       other       0.37      0.37      0.37        60
     pasture       0.30      0.60      0.40        47
        road       0.00      0.00      0.00         0
       water       0.97      0.85      0.90        33
    wetlands       1.00      0.45      0.62        31

   micro avg       0.74      0.74      0.74       600
   macro avg       0.58      0.53      0.54       600
weighted avg       0.79      0.74      0.76       600

Classification report for pixels:
              precision    recall  f1-score   support

    building       0.56      0.62      0.59   1585908
    corn_soy       0.88      0.76      0.82  12166991
   developed       0.68      0.61      0.64   6808575
      forest       0.73      0.78      0.75   5250299
       other       0.40      0.47      0.43   5173097
     pasture       0.22      0.35      0.27   3350083
        road       0.41      0.45      0.43    916645
       water       0.94      0.87      0.91   2026787
    wetlands       0.74      0.37      0.49   2043215

    accuracy                           0.64  39321600
   macro avg       0.62      0.59      0.59  39321600
weighted avg       0.68      0.64      0.65  39321600
```