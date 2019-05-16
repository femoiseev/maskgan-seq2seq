#!/usr/bin/env bash
fairseq-generate ../../data/processed/vocalization \
    --user-dir ../../models --task mask_mle \
    --path ../../checkpoints/mle_transformer_vocalization_stupid1/checkpoint_last.pt \
    --batch-size 512 --beam 1 --remove-bpe --raw-text \
    --quiet --log-format=tqdm | tee ../../results/mle_transformer_vocalization_stupid1.txt