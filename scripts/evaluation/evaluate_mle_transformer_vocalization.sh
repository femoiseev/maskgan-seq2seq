#!/usr/bin/env bash
fairseq-generate ../../data/processed/vocalization \
    --user-dir ../../models --task mask_mle \
    --path ../../checkpoints/mle_transformer_vocalization/checkpoint_best.pt \
    --batch-size 512 --beam 5 --remove-bpe --raw-text \
    --quiet --log-format=tqdm | tee ../../results/mle_transformer_vocalization.txt