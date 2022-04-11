#!/bin/bash

#cd ~
#cd ..
#cd ..
#cd ..
#cd /net/projects/scratch/winter/valid_until_31_July_2022/fheitzer/BAThesis-code/scripts
#bash runscript.sh -n Test -r 30 -e 1 -b 1 -c 1 -d 1000

name="Test"
rotation=30
epochs=1
batch_size=1
cycles=1
data_per_cycle=1000
bash runscript.sh "$name" $rotation $epochs $batch_size $cycles $data_per_cycle
