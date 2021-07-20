#!/bin/sh

python convert_dti2opennre.py
cd ../benchmark/dti
rm train.json
rm valid.json
rm test.json
