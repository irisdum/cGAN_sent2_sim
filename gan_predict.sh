#!/bin/bash

# first parameter is a path to the graph xml
model_path="$1"

# second parameter is a path to a parameter file
train_nber="$2"

weight="$3"

dataset="$4"

pref="$5"

path_csv="$6"

source /home/idumeur/miniconda3/etc/profile.d/conda.sh
conda activate training_env

export LD_LIBRARY_PATH=/home/idumeur/miniconda3/lib:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=/home/idumeur/miniconda3/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/idumeur/login/miniconda3/include:$CPLUS_INCLUDE_PATH

cp /srv/osirim/idumeur/data/dataset6/prepro1/input_large_dataset.zip  /tmp
unzip /tmp/input_large_dataset.zip -d /tmp


python predict.py --model_path ${model_path}   --tr_nber ${train_nber} --dataset ${dataset} --pref ${pref} --weights ${weight}

