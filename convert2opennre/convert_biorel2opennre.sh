#!/bin/sh

cd ../biorel
mv relation2id.json biorel_rel2id.json
cd ../convert2opennre 
python convert_biorel2opennre.py
cd ../biorel
rm train.json
rm dev.json
rm test.json