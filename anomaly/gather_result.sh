#!/bin/bash
set -e

MIX_MASK_RESULT_PATH="result/overlap_mixture"



echo "mix masked ====================="

echo "valve id_00"
grep AUC_mask_id_00 ${MIX_MASK_RESULT_PATH}/*valve* | awk '{print $3}'
echo "valve id_02"
grep AUC_mask_id_02 ${MIX_MASK_RESULT_PATH}/*valve* | awk '{print $3}'
echo "valve id_04"
grep AUC_mask_id_04 ${MIX_MASK_RESULT_PATH}/*valve* | awk '{print $3}'
echo "valve id_06"
grep AUC_mask_id_06 ${MIX_MASK_RESULT_PATH}/*valve* | awk '{print $3}'
echo "slider id_00"
grep AUC_mask_id_00 ${MIX_MASK_RESULT_PATH}/*slider* | awk '{print $3}'
echo "slider id_02"
grep AUC_mask_id_02 ${MIX_MASK_RESULT_PATH}/*slider* | awk '{print $3}'
echo "slider id_04"
grep AUC_mask_id_04 ${MIX_MASK_RESULT_PATH}/*slider* | awk '{print $3}'
echo "slider id_06"
grep AUC_mask_id_06 ${MIX_MASK_RESULT_PATH}/*slider* | awk '{print $3}'