#!/bin/bash
# -*- coding: utf-8 -*-

set -e

pwd=`dirname "$(readlink -f "$0")"`
base=$pwd/../../..
src=fr
tgt=en
data=$base/data/$tgt-$src/

tune=tune--model

# change into base directory to ensure paths are valid
cd $base

# create preprocessed directory
mkdir -p $data/$tune/

for num_layers in 1 2 3
do 
    mkdir -p $data/$tune/num_layers_$num_layers
    python train.py \
        --cuda \
        --data $data/prepared/ \
        --source-lang $src \
        --target-lang $tgt \
        --lr 0.003 \
        --batch-size 1 \
        --max-epoch 10000 \
        --patience 3 \
        --save-dir $data/$tune/num_layers_$num_layers \
         --restore-file checkpoint-last.pt \
        --save-interval 1 \
        --encoder-embed-dim 64 \
        --encoder-hidden-size 64 \
        --encoder-num-layers $num_layers \
        --encoder-bidirectional True \
        --encoder-dropout-in 0.25 \
        --encoder-dropout-out 0.25 \
        --decoder-embed-dim 64 \
        --decoder-hidden-size 128 \
        --decoder-num-layers $num_layers \
        --decoder-dropout-in 0.25 \
        --decoder-dropout-out 0.25
done

echo "train done!"
for num_layers in 1 2 3
do
    python translate.py \
        --cuda \
        --data $data/prepared/ \
        --dicts $data/prepared/ \
        --checkpoint-path $data/$tune/num_layers_$num_layers/checkpoint_last.pt \
        --output $data/$tune/num_layers_$num_layers.$tgt.txt
done


echo "translate done!"

for num_layers in 1 2 3
do
    bash scripts/postprocess.sh \
        $data/$tune/num_layers_$num_layers.$tgt.txt \
        $data/$tune/num_layers_$num_layers.$tgt.p.txt en
    cat $data/$tune/num_layers_$num_layers.$tgt.p.txt | sacrebleu $data/raw/test.$tgt
done