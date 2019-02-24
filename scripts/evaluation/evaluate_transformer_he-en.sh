#!/usr/bin/env bash
fairseq-generate ../../data/processed/he-en \
    --user-dir ../../models \
    --path ../../checkpoints/transformer_he-en/checkpoint_best.pt \
    --batch-size 512 --beam 5 --remove-bpe --raw-text \
    --quiet --log-format=tqdm | tee ../../results/transformer_he-en.txt