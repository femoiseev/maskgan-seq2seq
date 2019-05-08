#!/usr/bin/env bash
mkdir -p ../../checkpoints/mle_transformer_vocalization_stupid
CUDA_VISIBLE_DEVICES=0 fairseq-train ../../data/processed/vocalization \
  --user-dir ../../models --task mask_mle --raw-text \
  -a mask_transformer_vocalization --optimizer adam --lr 0.001 -s he -t voc \
  --label-smoothing 0.1 --dropout 0.3 --max-tokens 4000 \
  --min-lr '1e-06' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
  --warmup-updates 2500 \
  --criterion label_smoothed_cross_entropy \
  --adam-betas '(0.9, 0.98)' --save-dir ../../checkpoints/mle_transformer_vocalization_stupid \
  --keep-last-epochs 100 --max-epoch 50