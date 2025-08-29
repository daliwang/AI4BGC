#!/usr/bin/env bash
AI4BGC_DIR=/mnt/proj-shared/AI4BGC_7xw/

python ${AI4BGC_DIR}/AI4BGC/scripts/update_restart_with_aiprediction.py \
  --variable-list ${AI4BGC_DIR}/AI4BGC/CNP_IO_default.txt \
  --input-nc ${AI4BGC_DIR}/AI4BGC/ELM_data/original_20250408_trendytest_ICB1850CNPRDCTCBC.elm.r.0021-01-01-00000.nc \
  --output-nc ./restart_file_run_20250807_231637_updated.nc \
  --model-path ./cnp_predictions/model.pth \
  --data-paths ${AI4BGC_DIR}/TrainingData/Trendy_1_data_CNP/enhanced_dataset   \
  --inference-all