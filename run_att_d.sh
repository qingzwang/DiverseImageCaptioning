#! /bin/sh

id="att_x$1_c$2_d$3" # x: cross-entropy, c: CIDEr, d: discriminability


ckpt_path="../DiscCaptioning/log_RL_based/log_"$id

if [ ! -d $ckpt_path ]; then
  bash scripts/copy_model.sh att $id
fi

start_from="--start_from "$ckpt_path

python train.py \
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
--cider_optimization $2 \
--CIDEr_weight $2 \
--XE_weight $1 \
--cached_tokens coco-train-idxs