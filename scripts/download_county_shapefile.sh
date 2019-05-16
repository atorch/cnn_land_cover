#!/bin/bash

mkdir -p county

cd county

wget ftp://ftp2.census.gov/geo/tiger/TIGER2017/COUNTY/tl_2017_us_county.zip --no-clobber

unzip -o tl_2017_us_county.zip
