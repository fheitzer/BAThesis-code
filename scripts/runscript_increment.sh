#!/bin/bash
<<'###'
name=''
rotation=''
epochs=''
batch_size=''
cycles=''
data_per_cycle=''

while getopts n:r:e:b:c:d: arg; do
    case $arg in
        n) name=$OPATARG;;
        r) rotation=$OPTARG;;
        e) epochs=$OPTARG;;
        b) batch_size=$OPTARG;;
        c) cycles=$OPTARG;;
        d) data_per_cycle=$OPTARG;;
    esac
done
###
name="$1"
rotation="$2"
epochs="$3"
batch_size="$4"
cycles="$5"
data_per_cycle="$6"
cd ~
cd ..
cd ..
cd ..
cd ..
source /net/projects/scratch/winter/valid_until_31_July_2022/fheitzer/miniconda3/etc/profile.d/conda.sh
conda activate ba
cd /net/projects/scratch/winter/valid_until_31_July_2022/fheitzer/BAThesis-code/scripts
python run_increment.py --name "$name" --rotation "$rotation" --epochs "$epochs" --batch_size "$batch_size" --cycles "$cycles" --data_per_cycle "$data_per_cycle"
