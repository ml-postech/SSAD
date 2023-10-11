#!/bin/bash
set -e

SEEDS=(1 2 3 4 5 6 7 8 9 10)
MACHINES=("valve" "slider")
SOURCE1=("id_00" "id_04")
SOURCE2=("id_02" "id_06")
GPU=2

export CUDA_VISIBLE_DEVICES=$GPU
PYTHON="python"
PYTHON_FILE="baseline_mix_masked.py"

for seed in ${SEEDS[@]}; do
   for MACHINE in ${MACHINES[@]}; do
      for IDX in ${!SOURCE1[@]}; do
         SRC1=${SOURCE1[IDX]}
         SRC2=${SOURCE2[IDX]}
         sed -i "s@^seed.*@seed: ${seed}@g" baseline.yaml

         RESULT_DIR="/hdd/hdd1/kjc/ssad/result_231012_mix_masked_fix02"
         mkdir -p ${RESULT_DIR}
         sed -i "s@^result_directory.*@result_directory: ${RESULT_DIR}@g" baseline.yaml

         RESULT_NAME="mixture_baseline_${MACHINE}_seed${seed}_${SRC1}_${SRC2}.yaml"
         sed -i "s@^result_file.*@result_file: ${RESULT_NAME}@g" baseline.yaml

         sed -i "s@^MACHINE =.*@MACHINE = '${MACHINE}'@g" ${PYTHON_FILE}
         sed -i "s@^S1 =.*@S1 = '${SRC1}'@g" ${PYTHON_FILE}
         sed -i "s@^S2 =.*@S2 = '${SRC2}'@g" ${PYTHON_FILE}
         
         $PYTHON ${PYTHON_FILE}
      done
   done
done
