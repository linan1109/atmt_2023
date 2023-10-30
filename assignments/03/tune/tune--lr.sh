#!/bin/bash
# -*- coding: utf-8 -*-

set -e

pwd=`dirname "$(readlink -f "$0")"`
base=$pwd/../../..
src=fr
tgt=en
data=$base/data/$tgt-$src/

tune=tune--lr

# change into base directory to ensure paths are valid
cd $base

# create preprocessed directory
mkdir -p $data/$tune/

# train model with different learning rates
for lr in 0.0001 0.0005 0.001 0.005 0.01
do 
    mkdir -p $data/$tune/lr_$lr
    python train.py \
        --cuda \
        --data $data/prepared/ \
        --source-lang $src \
        --target-lang $tgt \
        --lr $lr \
        --batch-size 1 \
        --max-epoch 10000 \
        --patience 3 \
        --save-dir $data/$tune/lr_$lr \
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


# translate test set with different learning rates
for lr in 0.0001 0.0005 0.001 0.005 0.01
do
    python translate.py \
        --cuda \
        --data $data/prepared/ \
        --dicts $data/prepared/ \
        --checkpoint-path $data/$tune/lr_$lr/checkpoint_last.pt \
        --output $data/$tune/lr_$lr.$tgt.txt
done

echo "translate done!"

# evaluate BLEU score
for lr in 0.0001 0.0005 0.001 0.005 0.01
do
    echo "---------------------------------"
    echo "lr_$lr"
    bash scripts/postprocess.sh \
        $data/$tune/lr_$lr.$tgt.txt \
        $data/$tune/lr_$lr.$tgt.p.txt en
    cat $data/$tune/lr_$lr.$tgt.p.txt | sacrebleu $data/raw/test.$tgt 
done