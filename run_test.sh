#!/bin/bash

readonly ALGORITHM=$1
readonly TARGET_FOLDER="./results/lenet_tanh_compress_50_2/"
readonly TARGET_FILE="lenet_${ALGORITHM}_save.csv"
readonly NUM_RUNS=11

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
            python3.10 lenet_compression.py -comp random -pop 20 -its 20 -up 51 -lo 1 -hp
            ;;
            "PSO")
            python3.10 lenet_compression.py -comp pso -pop 20 -its 20 -up 51 -lo 1 -hp
            ;;
            "GA")
            python3.10 lenet_compression.py -comp genetic -pop 12 -its 36 -up 51 -lo 1 -hp
            ;;
            "BH")
            python3.10 lenet_compression.py -comp bh -pop 20 -its 20 -up 51 -lo 1 -hp
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