#!/bin/bash
set -e

SEEDS=(1 2 3 4 5 6 7 8 9 10)
MACHINES=("valve" "slider")
GPU=0

export CUDA_VISIBLE_DEVICES=$GPU
PYTHON="python"
PYTHON_FILE="baseline_masked.py"

for seed in ${SEEDS[@]}; do
   for MACHINE in ${MACHINES[@]}; do
      sed -i "s@^seed.*@seed: ${seed}@g" baseline.yaml

      RESULT_DIR="/hdd/hdd1/kjc/ssad/result_231012_single_masked_fix"
      mkdir -p ${RESULT_DIR}
      sed -i "s@^result_directory.*@result_directory: ${RESULT_DIR}@g" baseline.yaml

      RESULT_NAME="oracle_baseline_${MACHINE}_seed${seed}.yaml"
      sed -i "s@^result_file.*@result_file: ${RESULT_NAME}@g" baseline.yaml

      sed -i "s@^MACHINE =.*@MACHINE = '${MACHINE}'@g" ${PYTHON_FILE}

      ${PYTHON} ${PYTHON_FILE}
   done
done
