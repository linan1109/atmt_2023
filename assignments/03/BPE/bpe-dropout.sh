#!/bin/bash
# -*- coding: utf-8 -*-

set -e

pwd=`dirname "$(readlink -f "$0")"`
base=$pwd/../../..
src=fr
tgt=en

# change into base directory to ensure paths are valid
cd $base

for dropout in 0.0 0.025 0.05 0.075 0.1
do
    echo "---------------------------------------------"
    echo "dropout $dropout"

    data=$base/data/$tgt-$src/bpe/dropout_$dropout

    # create preprocessed directory
    mkdir -p $data/preprocessed/

    # normalize and tokenize raw data
    cat $data/../../raw/train.$src | perl moses_scripts/normalize-punctuation.perl -l $src | perl moses_scripts/tokenizer.perl -l $src -a -q > $data/preprocessed/train.$src.p
    cat $data/../../raw/train.$tgt | perl moses_scripts/normalize-punctuation.perl -l $tgt | perl moses_scripts/tokenizer.perl -l $tgt -a -q > $data/preprocessed/train.$tgt.p

    # train truecase models
    perl moses_scripts/train-truecaser.perl --model $data/preprocessed/tm.$src --corpus $data/preprocessed/train.$src.p
    perl moses_scripts/train-truecaser.perl --model $data/preprocessed/tm.$tgt --corpus $data/preprocessed/train.$tgt.p

    # apply truecase models to splits
    cat $data/preprocessed/train.$src.p | perl moses_scripts/truecase.perl --model $data/preprocessed/tm.$src > $data/preprocessed/train.$src 
    cat $data/preprocessed/train.$tgt.p | perl moses_scripts/truecase.perl --model $data/preprocessed/tm.$tgt > $data/preprocessed/train.$tgt

    # prepare remaining splits with learned models
    for split in valid test tiny_train
    do
        cat $data/../../raw/$split.$src | perl moses_scripts/normalize-punctuation.perl -l $src | perl moses_scripts/tokenizer.perl -l $src -a -q | perl moses_scripts/truecase.perl --model $data/preprocessed/tm.$src > $data/preprocessed/$split.$src
        cat $data/../../raw/$split.$tgt | perl moses_scripts/normalize-punctuation.perl -l $tgt | perl moses_scripts/tokenizer.perl -l $tgt -a -q | perl moses_scripts/truecase.perl --model $data/preprocessed/tm.$tgt > $data/preprocessed/$split.$tgt
    done

    # remove tmp files
    rm $data/preprocessed/train.$src.p
    rm $data/preprocessed/train.$tgt.p

    # preprocess all files for model training
    python bpe-preprocess.py \
        --target-lang $tgt \
        --source-lang $src \
        --dest-dir $data/prepared/ \
        --train-prefix $data/preprocessed/train \
        --valid-prefix $data/preprocessed/valid \
        --test-prefix $data/preprocessed/test \
        --tiny-train-prefix $data/preprocessed/tiny_train \
        --threshold-src 1 \
        --threshold-tgt 1 \
        --num-words-src 4000 \
        --num-words-tgt 4000 \
        --bpe-dropout $dropout

    echo "preprocess done!"

    python train.py \
        --data $data/prepared/ \
        --source-lang $src \
        --target-lang $tgt \
        --save-dir $data \
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
        --checkpoint-path $data/checkpoint_last.pt \
        --output $data/translated.$tgt.txt

    # remove all Ġ in the translated file
    sed -i 's/Ġ//g' $data/translated.$tgt.txt

    echo "translate done!"
    echo "dropout $dropout"
    bash scripts/postprocess.sh \
        $data/translated.$tgt.txt \
        $data/translated.$tgt.p.txt en
    cat $data/translated.$tgt.p.txt | sacrebleu $data/../../raw/test.$tgt

done

