#!/bin/bash


# settings
readonly NUM_RUNS=11
readonly TARGET_FOLDERS=(
    "./results/lenet_tanh_compress_50_dynamic_gmm/"
    "./results/lenet_relu_compress_50_dynamic_gmm/"
    )

readonly ALGORITHMS=(
    "GA"
    "PSO"
    "BH"
)
readonly PROGRAM_FILE='lenet_compression.py'
readonly UP_RANGE=51
readonly LOW_RAGE=1

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
    for algorithm in ${ALGORITHMS[@]}
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
                    python3.10 $PROGRAM_FILE -comp random -pop 20 -its 20 -up $UP_RANGE -lo $LOW_RAGE -hp $LOAD_COMMAND -sf $run_folder
                    ;;
                    "PSO")
                    echo $run_folder
                    python3.10 $PROGRAM_FILE -comp pso -pop 20 -its 20 -up $UP_RANGE -lo $LOW_RAGE -hp $LOAD_COMMAND -sf $run_folder
                    ;;
                    "GA")
                    python3.10 $PROGRAM_FILE -comp genetic -pop 12 -its 36 -up $UP_RANGE -lo $LOW_RAGE -hp $LOAD_COMMAND -sf $run_folder
                    ;;
                    "BH")
                    python3.10 $PROGRAM_FILE -comp blackhole -pop 20 -its 20 -up $UP_RANGE -lo $LOW_RAGE -hp $LOAD_COMMAND -sf $run_folder
                    ;;
                    *)
                    echo -n "unknown algorithm ${algorithm}"
                    exit
                    ;;
                esac
                echo "$run_folder completed"
            fi
        done

    done

done
