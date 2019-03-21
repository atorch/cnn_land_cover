#!/bin/bash

mkdir -p roads

cd roads

for county in 55023 55103
do
    wget ftp://ftp2.census.gov/geo/tiger/TIGER2017/ROADS/tl_2017_${county}_roads.zip --no-clobber
done
