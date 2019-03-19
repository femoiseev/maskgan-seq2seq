#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 fairseq-train ../../data/processed/vocalization \
  --user-dir ../../models --task mask_mle --raw-text \
  -a mask_transformer_vocalization --optimizer adam --lr 0.0005 -s he -t voc \
  --label-smoothing 0.1 --dropout 0.3 --max-tokens 4000 \
  --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
  --criterion label_smoothed_cross_entropy --max-update 50000 \
  --warmup-updates 4000 --warmup-init-lr '1e-07' \
  --adam-betas '(0.9, 0.98)' --save-dir ../../checkpoints/mle_transformer_vocalization \
  --keep-last-epochs 100