#!/bin/sh

#cp -r log_$1 ../DiscCaptioning/log_RL_based/log_$2
cp -r log_$1 /mnt/data/qingzhong/RL_based_models/log_$2
#cd ../DiscCaptioning/log_RL_based/log_$2
cd /mnt/data/qingzhong/RL_based_models/log_$2
mv infos_$1-best.pkl infos_$2-best.pkl
mv infos_$1.pkl infos_$2.pkl
cd /mnt/scratch/qingzhong/Other_captioning_Models/DiscCaptioning-master
