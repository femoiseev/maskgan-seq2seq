#!/usr/bin/env bash
{
    fairseq-generate ../../data/processed/he-en \
        --user-dir ../../models \
        --path ../../checkpoints/transformer_he-en/checkpoint_best.pt \
        --batch-size 512 --beam 5 --remove-bpe --raw-text | tee /tmp/gen.out
} &> /dev/null

grep ^H /tmp/gen.out | cut -f3- > /tmp/gen.out.sys
grep ^T /tmp/gen.out | cut -f2- > /tmp/gen.out.ref
fairseq-score --sys /tmp/gen.out.sys --ref /tmp/gen.out.ref | tee ../../results/transformer_he-en.txt