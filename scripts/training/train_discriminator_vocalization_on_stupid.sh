#!/usr/bin/env bash
!mkdir -p ../../checkpoints/discriminator_vocalization_on_stupid
CUDA_VISIBLE_DEVICES=0 fairseq-train ../../data/processed/vocalization \
  --user-dir ../../models --task mask_discriminator --raw-text \
  -a mask_discriminator_vocalization --optimizer adam --lr 0.001 -s he -t voc \
  --dropout 0.3 --max-tokens 4000 \
  --min-lr '1e-06' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
  --warmup-updates 2500 \
  --criterion discriminator_loss --max-epoch 25 \
  --adam-betas '(0.9, 0.98)' --save-dir ../../checkpoints/discriminator_vocalization_on_stupid \
  --generator-path ../../checkpoints/mle_transformer_vocalization_stupid/checkpoint_best.pt --keep-last-epochs 100