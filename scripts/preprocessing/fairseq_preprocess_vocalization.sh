#!/usr/bin/env bash
fairseq-preprocess \
  --trainpref ../../data/interim/vocalization/train --validpref ../../data/interim/vocalization/valid --testpref ../../data/interim/vocalization/test \
  --source-lang he --target-lang voc \
  --destdir ../../data/processed/vocalization --output-format raw