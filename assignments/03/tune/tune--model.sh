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
        --data $data/prepared/ \
        --source-lang $src \
        --target-lang $tgt \
        --save-dir $data/$tune/num_layers_$num_layers \
        --encoder-num-layers $num_layers \
        --decoder-num-layers $num_layers 
done

echo "train done!"
for num_layers in 1 2 3
do
    python translate.py \
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
