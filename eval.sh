#! /bin/sh
id=$1
method=$3
num_samples=$(($4-1))
resdir=/mnt/data/qingzhong/results

if [ ! -d $resdir/log_$id ]; then
    mkdir $resdir/log_$id
fi

if [ ! -d $resdir/log_$id/$method ]; then
    mkdir $resdir/log_$id/$method
fi

if [ ! -d $resdir/log_$id/beam_search ]; then
    mkdir $resdir/log_$id/beam_search
fi

for VARIABLE in $(seq 0 $num_samples)
do
    python eval.py \
    --decoding_constraint 0 \
    --dump_images 0 \
    --num_images -1 \
    --batch_size 50 \
    --split $2  \
    --input_label_h5 data/cocotalk_label.h5 \
    --input_fc_dir ../data/cocotalk_fc \
    --input_att_dir ../data/cocobu_att \
    --model /mnt/data/qingzhong/RL_based_models/log_$id/model.pth \
    --language_eval 1 \
    --beam_size 0 \
    --temperature 1.0 \
    --sample_max 0 \
    --infos_path /mnt/data/qingzhong/RL_based_models/log_$id/infos_$id.pkl \
    --dump_file $resdir/log_$id/$method/results$VARIABLE.json
done

for VARIABLE in 3
do
    python eval.py \
    --decoding_constraint 0 \
    --dump_images 0 \
    --num_images -1 \
    --batch_size 50 \
    --split $2  \
    --input_label_h5 data/cocotalk_label.h5 \
    --input_fc_dir ../data/cocotalk_fc \
    --input_att_dir ../data/cocobu_att \
    --model /mnt/data/qingzhong/RL_based_models/log_$id/model.pth \
    --language_eval 1 \
    --beam_size $VARIABLE \
    --temperature 1.0 \
    --sample_max 1 \
    --infos_path /mnt/data/qingzhong/RL_based_models/log_$id/infos_$id.pkl \
    --dump_file $resdir/log_$id/beam_search/results_bs$VARIABLE.json
done
