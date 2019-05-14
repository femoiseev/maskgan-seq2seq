#!/usr/bin/env bash
mkdir -p ../../checkpoints/discriminator_vocalization
CUDA_VISIBLE_DEVICES=0 fairseq-train ../../data/processed/vocalization \
  --user-dir ../../models --task mask_discriminator --raw-text \
  -a mask_discriminator_vocalization --optimizer adam --lr 0.0005 -s he -t voc \
  --dropout 0.3 --max-tokens 4000 \
  --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
  --criterion discriminator_loss --max-epoch 1000 \
  --warmup-updates 4000 --warmup-init-lr '1e-07' \
  --adam-betas '(0.9, 0.98)' --save-dir ../../checkpoints/discriminator_vocalization \
  --generator-path ../../checkpoints/mle_transformer_vocalization/checkpoint_best.pt --keep-last-epochs 100