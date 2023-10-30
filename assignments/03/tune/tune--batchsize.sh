#!/bin/bash
# -*- coding: utf-8 -*-

set -e

pwd=`dirname "$(readlink -f "$0")"`
base=$pwd/../../..
src=fr
tgt=en
data=$base/data/$tgt-$src/

tune=tune--batchsize

# change into base directory to ensure paths are valid
cd $base

# create preprocessed directory
mkdir -p $data/$tune/

# train model with different learning rates
for bs in 16 32 64 128
do 
    mkdir -p $data/$tune/bs_$bs
    python train.py \
        --cuda \
        --data $data/prepared/ \
        --source-lang $src \
        --target-lang $tgt \
        --lr 0.003 \
        --batch-size $bs \
        --max-epoch 10000 \
        --patience 3 \
        --save-dir $data/$tune/bs_$bs \
        --restore-file checkpoint-last.pt \
        --save-interval 1 \
        --encoder-embed-dim 64 \
        --encoder-hidden-size 64 \
        --encoder-num-layers 1 \
        --encoder-bidirectional True \
        --encoder-dropout-in 0.25 \
        --encoder-dropout-out 0.25 \
        --decoder-embed-dim 64 \
        --decoder-hidden-size 128 \
        --decoder-num-layers 1 \
        --decoder-dropout-in 0.25 \
        --decoder-dropout-out 0.25
done

echo "train done!"

for bs in 16 32 64 128
do
    python translate.py \
        --cuda \
        --data $data/prepared/ \
        --dicts $data/prepared/ \
        --checkpoint-path $data/$tune/bs_$bs/checkpoint_last.pt \
        --output $data/$tune/bs_$bs.$tgt.txt
done


echo "translate done!"

for bs in 16 32 64 128
do
    bash scripts/postprocess.sh \
        $data/$tune/bs_$bs.$tgt.txt \
        $data/$tune/bs_$bs.$tgt.p.txt en
    cat $data/$tune/bs_$bs.$tgt.p.txt | sacrebleu $data/raw/test.$tgt
done