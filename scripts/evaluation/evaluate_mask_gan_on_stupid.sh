#!/usr/bin/env bash
fairseq-generate ../../data/processed/vocalization \
    --user-dir ../../models --task mask_gan \
    --path ../../checkpoints/mask_gan_vocalization_on_stupid/checkpoint_best.pt \
    --batch-size 512 --beam 5 --remove-bpe --raw-text \
    --quiet --log-format=tqdm | tee ../../results/mask_gan_vocalization_on_stupid.txt