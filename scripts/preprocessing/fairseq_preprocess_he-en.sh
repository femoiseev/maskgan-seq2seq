#!/usr/bin/env bash
fairseq-preprocess \
  --trainpref ../../data/interim/he-en/train --validpref ../../data/interim/he-en/valid --testpref ../../data/interim/he-en/test \
  --source-lang he --target-lang en \
  --destdir ../../data/processed/he-en --output-format raw