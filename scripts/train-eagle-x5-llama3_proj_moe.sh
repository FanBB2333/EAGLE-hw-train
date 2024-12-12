#!/bin/bash
export DATE_TODAY=`date '+%m%d'`
# echo $DATE_TODAY.txt
export OUTPUT_FOLDER=/home1/hxl/disk/EAGLE/output
# echo $OUTPUT_FOLDER

bash /home1/hxl/disk/EAGLE/scripts/pretrain-eagle-x5-llama3-8b.sh pretrain$DATE_TODAY >> $OUTPUT_FOLDER/pretrain/$DATE_TODAY.txt

bash /home1/hxl/disk/EAGLE/scripts/finetune-eagle-x5-llama3-8b-1.8m_moe.sh finetune$DATE_TODAY >> $OUTPUT_FOLDER/finetune/$DATE_TODAY.txt