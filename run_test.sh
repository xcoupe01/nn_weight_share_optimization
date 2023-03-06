#!/bin/bash

readonly ALGORITHM=$1
readonly TARGET_FOLDER="./results/lenet_tanh_compress_50_dynamic_normal/"
readonly TARGET_FILE="lenet_${ALGORITHM}_save.csv"
readonly NUM_RUNS=11
LOAD_COMMAND=""

mkdir -p "$TARGET_FOLDER"

# search for config to load, if not found create one
if [ -f "${TARGET_FOLDER}experiment_setting.yaml" ]
then
    LOAD_COMMAND=" -cfl ${TARGET_FOLDER}experiment_setting.yaml"
    echo 'experiments settings file found and loaded'
else  
    python3.10 lenet_compression.py -cfs "${TARGET_FOLDER}experiment_setting.yaml"
    echo 'experiments settings file created'
fi

# main loop
for i in $(seq 1 $NUM_RUNS)
do
    folder="${TARGET_FOLDER}run$i"
    mkdir -p "$folder"
    echo "processing folder $folder"
    if [ -f "$folder/$TARGET_FILE" ]
    then
        echo "${folder}/$TARGET_FILE already done"
    else
        case $ALGORITHM in
            "RND")
            python3.10 lenet_compression.py -comp random -pop 20 -its 20 -up 51 -lo 1 -hp $LOAD_COMMAND
            ;;
            "PSO")
            python3.10 lenet_compression.py -comp pso -pop 20 -its 20 -up 51 -lo 1 -hp $LOAD_COMMAND
            ;;
            "GA")
            python3.10 lenet_compression.py -comp genetic -pop 12 -its 36 -up 51 -lo 1 -hp $LOAD_COMMAND
            ;;
            "BH")
            python3.10 lenet_compression.py -comp blackhole -pop 20 -its 20 -up 51 -lo 1 -hp $LOAD_COMMAND
            ;;
            *)
            echo -n "unknown algorithm"
            exit
            ;;
        esac
        mv "./results/$TARGET_FILE" "$folder/$TARGET_FILE"
        echo "$folder completed"

    fi
done