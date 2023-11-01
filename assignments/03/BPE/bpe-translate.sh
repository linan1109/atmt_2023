#!/bin/bash
# -*- coding: utf-8 -*-

set -e

pwd=`dirname "$(readlink -f "$0")"`
base=$pwd/../../..
src=fr
tgt=en
data=$base/data/$tgt-$src/bpe/

# change into base directory to ensure paths are valid
cd $base

python train.py \
    --cuda \
    --data $data/prepared/ \
    --source-lang $src \
    --target-lang $tgt \
    --save-dir $data\
    --train-on-tiny \
    --batch-size 32

echo "train done!"

python translate.py \
    --cuda \
    --data $data/prepared/ \
    --dicts $data/prepared/ \
    --checkpoint-path $data/checkpoint_last.pt \
    --output $data/translated.$tgt.txt

# remove all Ġ in the translated file
sed -i 's/Ġ//g' $data/translated.$tgt.txt

echo "translate done!"

bash scripts/postprocess.sh \
    $data/translated.$tgt.txt \
    $data/translated.$tgt.p.txt en
cat $data/translated.$tgt.p.txt | sacrebleu $data/../raw/test.$tgt
