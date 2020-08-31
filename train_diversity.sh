#! /bin/sh
# x: cross-entropy, c: CIDEr, d: discriminability, div: diversity, LSA/selfcider/mcider, naiveRL: 0 or 1, self-critical: 0/1
id="att_x$1_c$2_d$3_div$4_$5_naiverRL_$6_numC$7"
#_naiverRL_$6 _self-critical$7


#ckpt_path="../DiscCaptioning/log_RL_based/log_"$id

ckpt_path="/mnt/data/qingzhong/RL_based_models/log_"$id

if [ ! -d $ckpt_path ]; then
  bash scripts/copy_model.sh att $id
fi

start_from="--start_from "$ckpt_path

CUDA_VISIBLE_DEVICES=1 python train_diversity.py \
--id $id \
--caption_model att2in2 \
--vse_model fc \
--share_embed 0 \
--input_json data/cocotalk.json \
--input_label_h5 data/cocotalk_label.h5 \
--input_fc_dir ../data/cocotalk_fc \
--input_att_dir ../data/cocobu_att \
--batch_size 128 \
--seq_per_img 1 \
--beam_size 3 \
--learning_rate 5e-4 \
--learning_rate_decay_start 0 \
--learning_rate_decay_every 15 \
--scheduled_sampling_start 0 \
--checkpoint_path $ckpt_path $start_from \
--save_checkpoint_every 3000 \
--language_eval 1 \
--val_images_use 5000 \
--max_epochs 210 \
--retrieval_reward reinforce \
--retrieval_reward_weight $3 \
--vse_loss_weight 0 \
--caption_loss_weight 1 \
--initialize_retrieval log_fc_con/model_vse-best.pth \
--cider_optimization 0 \
--CIDEr_weight $2 \
--XE_weight $1 \
--Div_weight $4 \
--num_sample_captions $7 \
--diversity_metric $5 \
--naive_RL $6 \
--cached_tokens coco-train-idxs
