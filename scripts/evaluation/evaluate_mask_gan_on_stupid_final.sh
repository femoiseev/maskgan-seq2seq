#!/usr/bin/env bash
fairseq-generate ../../data/processed/vocalization \
    --user-dir ../../models --task mask_gan \
    --path ../../checkpoints/mask_gan_vocalization_on_stupid_final/checkpoint35.pt \
    --batch-size 512 --beam 1 --remove-bpe --raw-text \
    --quiet --log-format=tqdm | tee ../../results/mask_gan_vocalization_on_stupid_final.txt