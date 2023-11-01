#!/bin/bash
# -*- coding: utf-8 -*-

set -e

pwd=`dirname "$(readlink -f "$0")"`
base=$pwd/../../..
src=fr
tgt=en
data=$base/data/$tgt-$src/

tune=tune--dropout
# change into base directory to ensure paths are valid
cd $base

# create preprocessed directory
mkdir -p $data/$tune/

for dropout in 0 0.3 0.5 0.7
do
    mkdir -p $data/$tune/dropout_$dropout
    python train.py \
        --data $data/prepared/ \
        --source-lang $src \
        --target-lang $tgt \
        --save-dir $data/$tune/dropout_$dropout \
        --encoder-dropout-in $dropout \
        --encoder-dropout-out $dropout \
        --decoder-dropout-in $dropout \
        --decoder-dropout-out $dropout
done 

echo "train done!"

for dropout in 0 0.3 0.5 0.7
do
    python translate.py \
        --data $data/prepared/ \
        --dicts $data/prepared/ \
        --checkpoint-path $data/$tune/dropout_$dropout/checkpoint_last.pt \
        --output $data/$tune/dropout_$dropout.$tgt.txt
done

echo "translate done!"

for dropout in 0 0.3 0.5 0.7
do
    bash scripts/postprocess.sh \
        $data/$tune/dropout_$dropout.$tgt.txt \
        $data/$tune/dropout_$dropout.$tgt.p.txt en
    cat $data/$tune/dropout_$dropout.$tgt.p.txt | sacrebleu $data/raw/test.$tgt
done
