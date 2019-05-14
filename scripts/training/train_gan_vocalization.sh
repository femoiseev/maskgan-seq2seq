#!/usr/bin/env bash
mkdir -p ../../checkpoints/gan_vocalization

CUDA_VISIBLE_DEVICES=0 fairseq-train ../../data/processed/vocalization \
  --user-dir ../../models --task mask_gan --raw-text \
  -a mask_transformer_vocalization --optimizer adam --lr 0.0003 -s he -t voc \
  --dropout 0.3 --max-tokens 4000 \
  --min-lr '1e-06' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
  --reset-optimizer \
  --warmup-updates 2500 \
  --criterion generator_loss --max-epoch 200 \
  --discriminator-steps 3 \
  --adam-betas '(0.9, 0.98)' --save-dir ../../checkpoints/gan_generator \
  --discriminator-path ../../checkpoints/discriminator_vocalization/checkpoint_best.pt --keep-last-epochs 100