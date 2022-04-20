#!/bin/bash

#cd ~
#cd ..
#cd ..
#cd ..
#cd /net/projects/scratch/winter/valid_until_31_July_2022/fheitzer/BAThesis-code/scripts
#bash runscript.sh -n Test -r 30 -e 1 -b 1 -c 1 -d 1000

name="Frozenmodel_Increment"
rotation=90
cycles=90
bash runscript_frozenmodel_increment.sh "$name" $rotation $cycles
