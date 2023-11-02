#!/bin/bash
# -*- coding: utf-8 -*-

set -e

pwd=`dirname "$(readlink -f "$0")"`
base=$pwd/../../..
src=fr
tgt=en
data=$base/data/$tgt-$src/
tune=tune--optimal
# change into base directory to ensure paths are valid
cd $base

mkdir -p $data/$tune
python train.py \
    --data $data/prepared/ \
    --source-lang $src \
    --target-lang $tgt \
    --save-dir $data/$tune \
    --lr 0.0005 \
    --batch-size 8 \
    --encoder-num-layers 2 \
    --decoder-num-layers 2 \
    --encoder-dropout-in 0.3 \
    --encoder-dropout-out 0.3 \
    --decoder-dropout-in 0.3 \
    --decoder-dropout-out 0.3

echo "train done!"

python translate.py \
    --data $data/prepared/ \
    --dicts $data/prepared/ \
    --checkpoint-path $data/$tune/checkpoint_last.pt \
    --output $data/$tune/$tgt.txt

echo "translate done!"

bash scripts/postprocess.sh \
    $data/$tune/$tgt.txt \
    $data/$tune/$tgt.p.txt en
cat $data/$tune/$tgt.p.txt | sacrebleu $data/raw/test.$tgt 
