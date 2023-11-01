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
for bs in 1 8 16 32 64 128
do 
    mkdir -p $data/$tune/bs_$bs
    python train.py \
        --data $data/prepared/ \
        --source-lang $src \
        --target-lang $tgt \
        --batch-size $bs \
        --save-dir $data/$tune/bs_$bs 
done

echo "train done!"

for bs in 1 8 16 32 64 128
do
    python translate.py \
        --data $data/prepared/ \
        --dicts $data/prepared/ \
        --checkpoint-path $data/$tune/bs_$bs/checkpoint_last.pt \
        --output $data/$tune/bs_$bs.$tgt.txt
done


echo "translate done!"

for bs in 1 8 16 32 64 128
do
    bash scripts/postprocess.sh \
        $data/$tune/bs_$bs.$tgt.txt \
        $data/$tune/bs_$bs.$tgt.p.txt en
    cat $data/$tune/bs_$bs.$tgt.p.txt | sacrebleu $data/raw/test.$tgt
done
