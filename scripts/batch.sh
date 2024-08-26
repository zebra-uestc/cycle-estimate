#!/bin/bash

root_dir=${CYCLE_ESTIMATE_DIR}
benchmarks=("perlbench_checkspam" "bzip2_chicken" "gcc_166" "bwaves" "gamess_cytosine" "milc")

function process(){
    for benchmark in "${benchmarks[@]}"; do
        mkdir -p ${root_dir}/logs/${benchmark}
        python ${root_dir}/src/main.py ${benchmark} > ${root_dir}/logs/${benchmark}/main.log 2>&1 
        echo "process ${benchmark} done."
    done
}
export -f process

function rm_preds(){
    for benchmark in "${benchmarks[@]}"; do
        file=${root_dir}/data/${benchmark}/total_cycle_pred.txt
        rm $file
        echo "remove ${file} done."
    done
}

export -f rm_preds

process