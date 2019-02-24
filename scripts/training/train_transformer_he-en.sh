#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 fairseq-train ../../data/processed/he-en \
  --user-dir ../../models --task translation --raw-text \
  -a transformer_he-en --optimizer adam --lr 0.0005 -s he -t en \
  --label-smoothing 0.1 --max-tokens 4000 \
  --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
  --criterion label_smoothed_cross_entropy --me 200 \
  --warmup-updates 4000 --warmup-init-lr '1e-07' \
  --adam-betas '(0.9, 0.98)' --save-dir ../../checkpoints/transformer_he-en \
  --keep-last-epochs 100