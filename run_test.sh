#!/bin/bash

readonly NUM_RUNS=11
readonly TARGET_FOLDERS=(
    "./results/mobnet_compress_2N_120_dynamic_mbk/"
    )

readonly ALGORITHMS=(
    "GA"
)

LOAD_COMMAND=""

# main loop
for folder in ${TARGET_FOLDERS[@]}
do

    mkdir -p "$folder"

    # search for config to load, if not found create one
    if [ -f "${folder}experiment_setting.yaml" ]
    then
        LOAD_COMMAND=" -cfl ${folder}experiment_setting.yaml"
        echo 'experiments settings file found and loaded'
    else  
        python3.10 net_compression.py -cfs "${folder}experiment_setting.yaml"
        echo 'experiments settings file created'
    fi

    # do all algorithms
    for algorithm in ${ALGORITHMS}
    do
        TARGET_FILE="${algorithm}_save.csv"
        
        # do all runs
        for i in $(seq 1 $NUM_RUNS)
        do
            run_folder="${folder}run$i"
            mkdir -p "$run_folder"
            echo "processing folder $run_folder"
            if [ -f "$run_folder/$TARGET_FILE" ]
            then
                echo "${run_folder}/$TARGET_FILE already done"
            else
                case $algorithm in
                    "RND")
                    python3.10 net_compression.py -comp random -pop 20 -its 20 -up 51 -lo 1 -hp $LOAD_COMMAND
                    ;;
                    "PSO")
                    python3.10 net_compression.py -comp pso -pop 20 -its 20 -up 51 -lo 1 -hp $LOAD_COMMAND
                    ;;
                    "GA")
                    python3.10 net_compression.py -comp genetic -pop 12 -its 36 -up 51 -lo 1 -hp $LOAD_COMMAND
                    ;;
                    "BH")
                    python3.10 net_compression.py -comp blackhole -pop 20 -its 20 -up 51 -lo 1 -hp $LOAD_COMMAND
                    ;;
                    *)
                    echo -n "unknown algorithm ${algorithm}"
                    exit
                    ;;
                esac
                mv "./results/$TARGET_FILE" "$run_folder/$TARGET_FILE"
                echo "$run_folder completed"
            fi
        done

    done

done
