#!/bin/bash
set -e

BASE_RESULT_PATH="/hdd/hdd1/kjc/ssad/result_231007_single"
BASE_MASK_RESULT_PATH="/hdd/hdd1/kjc/ssad/result_231012_single_masked_fix"
MIX_RESULT_PATH="/hdd/hdd1/kjc/ssad/result_231016_overlap_mix"
MIX_MASK_RESULT_PATH="/hdd/hdd1/kjc/ssad/result_231016_overlap_mix_masked"
SEP_RESULT_PATH="/hdd/hdd1/kjc/ssad/result_231016_overlap_sep_no_mask"
SEP_MASK_RESULT_PATH="/hdd/hdd1/kjc/ssad/result_231016_overlap_ssad"

echo "single ====================="
echo "valve id_00"
grep -A 2 id_00 ${BASE_RESULT_PATH}/oracle_*valve* | grep AUC_mean | awk '{print $3}'
echo "valve id_02"
grep -A 2 id_02 ${BASE_RESULT_PATH}/oracle_*valve* | grep AUC_mean | awk '{print $3}'
echo "valve id_04"
grep -A 2 id_04 ${BASE_RESULT_PATH}/oracle_*valve* | grep AUC_mean | awk '{print $3}'
echo "valve id_06"
grep -A 2 id_06 ${BASE_RESULT_PATH}/oracle_*valve* | grep AUC_mean | awk '{print $3}'

echo "slider id_00"
grep -A 2 id_00 ${BASE_RESULT_PATH}/oracle_*slider* | grep AUC_mean | awk '{print $3}'
echo "slider id_02"
grep -A 2 id_02 ${BASE_RESULT_PATH}/oracle_*slider* | grep AUC_mean | awk '{print $3}'
echo "slider id_04"
grep -A 2 id_04 ${BASE_RESULT_PATH}/oracle_*slider* | grep AUC_mean | awk '{print $3}'
echo "slider id_06"
grep -A 2 id_06 ${BASE_RESULT_PATH}/oracle_*slider* | grep AUC_mean | awk '{print $3}'

echo "single masked ====================="

echo "valve id_00"
grep -A 2 id_00 ${BASE_MASK_RESULT_PATH}/oracle_*valve* | grep AUC_mask | awk '{print $3}'
echo "valve id_02"
grep -A 2 id_02 ${BASE_MASK_RESULT_PATH}/oracle_*valve* | grep AUC_mask | awk '{print $3}'
echo "valve id_04"
grep -A 2 id_04 ${BASE_MASK_RESULT_PATH}/oracle_*valve* | grep AUC_mask | awk '{print $3}'
echo "valve id_06"
grep -A 2 id_06 ${BASE_MASK_RESULT_PATH}/oracle_*valve* | grep AUC_mask | awk '{print $3}'

echo "slider id_00"
grep -A 2 id_00 ${BASE_MASK_RESULT_PATH}/oracle_*slider* | grep AUC_mask | awk '{print $3}'
echo "slider id_02"
grep -A 2 id_02 ${BASE_MASK_RESULT_PATH}/oracle_*slider* | grep AUC_mask | awk '{print $3}'
echo "slider id_04"
grep -A 2 id_04 ${BASE_MASK_RESULT_PATH}/oracle_*slider* | grep AUC_mask | awk '{print $3}'
echo "slider id_06"
grep -A 2 id_06 ${BASE_MASK_RESULT_PATH}/oracle_*slider* | grep AUC_mask | awk '{print $3}'

echo "mix ====================="

echo "valve id_00"
grep AUC_mean_id_00 ${MIX_RESULT_PATH}/*valve* | awk '{print $3}'
echo "valve id_02"
grep AUC_mean_id_02 ${MIX_RESULT_PATH}/*valve* | awk '{print $3}'
echo "valve id_04"
grep AUC_mean_id_04 ${MIX_RESULT_PATH}/*valve* | awk '{print $3}'
echo "valve id_06"
grep AUC_mean_id_06 ${MIX_RESULT_PATH}/*valve* | awk '{print $3}'
echo "slider id_00"
grep AUC_mean_id_00 ${MIX_RESULT_PATH}/*slider* | awk '{print $3}'
echo "slider id_02"
grep AUC_mean_id_02 ${MIX_RESULT_PATH}/*slider* | awk '{print $3}'
echo "slider id_04"
grep AUC_mean_id_04 ${MIX_RESULT_PATH}/*slider* | awk '{print $3}'
echo "slider id_06"
grep AUC_mean_id_06 ${MIX_RESULT_PATH}/*slider* | awk '{print $3}'


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


echo "sep ====================="

echo "valve id_00"
grep AUC_mean_id_00 ${SEP_RESULT_PATH}/*valve* | awk '{print $3}'
echo "valve id_02"
grep AUC_mean_id_02 ${SEP_RESULT_PATH}/*valve* | awk '{print $3}'
echo "valve id_04"
grep AUC_mean_id_04 ${SEP_RESULT_PATH}/*valve* | awk '{print $3}'
echo "valve id_06"
grep AUC_mean_id_06 ${SEP_RESULT_PATH}/*valve* | awk '{print $3}'
echo "slider id_00"
grep AUC_mean_id_00 ${SEP_RESULT_PATH}/*slider* | awk '{print $3}'
echo "slider id_02"
grep AUC_mean_id_02 ${SEP_RESULT_PATH}/*slider* | awk '{print $3}'
echo "slider id_04"
grep AUC_mean_id_04 ${SEP_RESULT_PATH}/*slider* | awk '{print $3}'
echo "slider id_06"
grep AUC_mean_id_06 ${SEP_RESULT_PATH}/*slider* | awk '{print $3}'

echo "ssad ====================="

echo "valve id_00"
grep AUC_mask_id_00 ${SEP_MASK_RESULT_PATH}/*valve* | awk '{print $3}'
echo "valve id_02"
grep AUC_mask_id_02 ${SEP_MASK_RESULT_PATH}/*valve* | awk '{print $3}'
echo "valve id_04"
grep AUC_mask_id_04 ${SEP_MASK_RESULT_PATH}/*valve* | awk '{print $3}'
echo "valve id_06"
grep AUC_mask_id_06 ${SEP_MASK_RESULT_PATH}/*valve* | awk '{print $3}'
echo "slider id_00"
grep AUC_mask_id_00 ${SEP_MASK_RESULT_PATH}/*slider* | awk '{print $3}'
echo "slider id_02"
grep AUC_mask_id_02 ${SEP_MASK_RESULT_PATH}/*slider* | awk '{print $3}'
echo "slider id_04"
grep AUC_mask_id_04 ${SEP_MASK_RESULT_PATH}/*slider* | awk '{print $3}'
echo "slider id_06"
grep AUC_mask_id_06 ${SEP_MASK_RESULT_PATH}/*slider* | awk '{print $3}'
