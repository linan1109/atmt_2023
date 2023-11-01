import argparse
import collections
import logging
import os
import sys
import re
import pickle

# establish link to seq2seq dir
# scripts_dir = os.path.dirname(os.path.abspath(__file__))
# base_dir = os.path.join(scripts_dir, "..")
# sys.path.append(base_dir)

from seq2seq import utils
from seq2seq.data.dictionary import Dictionary

SPACE_NORMALIZER = re.compile("\s+")

def line_split(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    words = line.split()
    words = [word + "Ġ" for word in words]
    return words

def get_args():
    parser = argparse.ArgumentParser('Data pre-processing)')
    parser.add_argument('--source-lang', default=None, metavar='SRC', help='source language')
    parser.add_argument('--target-lang', default=None, metavar='TGT', help='target language')

    parser.add_argument('--train-prefix', default=None, metavar='FP', help='train file prefix')
    parser.add_argument('--tiny-train-prefix', default=None, metavar='FP', help='tiny train file prefix')
    parser.add_argument('--valid-prefix', default=None, metavar='FP', help='valid file prefix')
    parser.add_argument('--test-prefix', default=None, metavar='FP', help='test file prefix')
    parser.add_argument('--dest-dir', default='data-bin', metavar='DIR', help='destination dir')

    parser.add_argument('--threshold-src', default=2, type=int,
                        help='map words appearing less than threshold times to unknown')
    parser.add_argument('--num-words-src', default=-1, type=int, help='number of source words to retain')
    parser.add_argument('--threshold-tgt', default=2, type=int,
                        help='map words appearing less than threshold times to unknown')
    parser.add_argument('--num-words-tgt', default=-1, type=int, help='number of target words to retain')
    parser.add_argument('--vocab-src', default=None, type=str, help='path to dictionary')
    parser.add_argument('--vocab-trg', default=None, type=str, help='path to dictionary')
    parser.add_argument('--quiet', action='store_true', help='no logging')

    return parser.parse_args()


def main(args):
    os.makedirs(args.dest_dir, exist_ok=True)
    if not args.vocab_src:
        src_dict = build_dictionary([args.train_prefix + '.' + args.source_lang], args.num_words_src, os.path.join(args.dest_dir, 'merge.' + args.source_lang))

        src_dict.finalize(threshold=args.threshold_src, num_words=args.num_words_src)
        src_dict.save(os.path.join(args.dest_dir, 'dict.' + args.source_lang))
        if not args.quiet:
            logging.info('Built a source dictionary ({}) with {} words'.format(args.source_lang, len(src_dict)))

    else:
        src_dict = Dictionary.load(args.vocab_src)
        if not args.quiet:
            logging.info('Loaded a source dictionary ({}) with {} words'.format(args.source_lang, len(src_dict)))

    if not args.vocab_trg:
        tgt_dict = build_dictionary([args.train_prefix + '.' + args.target_lang], args.num_words_src, os.path.join(args.dest_dir, 'merge.' + args.target_lang))

        tgt_dict.finalize(threshold=args.threshold_tgt, num_words=args.num_words_tgt)
        tgt_dict.save(os.path.join(args.dest_dir, 'dict.' + args.target_lang))
        if not args.quiet:
            logging.info('Built a target dictionary ({}) with {} words'.format(args.target_lang, len(tgt_dict)))

    else:
        tgt_dict = Dictionary.load(args.vocab_trg)
        if not args.quiet:
            logging.info('Loaded a target dictionary ({}) with {} words'.format(args.target_lang, len(tgt_dict)))

    def make_split_datasets(lang, dictionary):
        if args.train_prefix is not None:
            make_binary_dataset(args.train_prefix + '.' + lang, os.path.join(args.dest_dir, 'train.' + lang),
                                dictionary, merge_dict=os.path.join(args.dest_dir, 'merge.' + lang))
        if args.tiny_train_prefix is not None:
            make_binary_dataset(args.tiny_train_prefix + '.' + lang, os.path.join(args.dest_dir, 'tiny_train.' + lang),
                                dictionary, merge_dict=os.path.join(args.dest_dir, 'merge.' + lang))
        if args.valid_prefix is not None:
            make_binary_dataset(args.valid_prefix + '.' + lang, os.path.join(args.dest_dir, 'valid.' + lang),
                                dictionary, merge_dict=os.path.join(args.dest_dir, 'merge.' + lang))
        if args.test_prefix is not None:
            make_binary_dataset(args.test_prefix + '.' + lang, os.path.join(args.dest_dir, 'test.' + lang), 
                                dictionary, merge_dict=os.path.join(args.dest_dir, 'merge.' + lang))

    make_split_datasets(args.source_lang, src_dict)
    make_split_datasets(args.target_lang, tgt_dict)

def compute_pair_freqs(splits, word_freqs):
    pair_freqs = collections.defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs

def merge_pair(a, b, splits, word_freqs):
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue

        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits

def build_dictionary(filenames, vocab_size, merge_dict):
    dictionary = Dictionary()
    all_sents = []
    for filename in filenames:
        with open(filename, 'r') as file:
            for line in file:
                all_sents.append(line)
    
    word_freqs = collections.defaultdict(int)
    for text in all_sents:
        words = line_split(text)
        for word in words:
            word_freqs[word] += 1

    vocab = collections.defaultdict(int)
    for word in word_freqs.keys():
        for letter in word:
            vocab[letter] += word_freqs[word]

    splits = {word: [c for c in word] for word in word_freqs.keys()}

    merges = {}
    while len(vocab) < vocab_size:
        pair_freqs = compute_pair_freqs(splits, word_freqs)
        best_pair = ""
        max_freq = None
        for pair, freq in pair_freqs.items():
            if max_freq is None or max_freq < freq:
                best_pair = pair
                max_freq = freq
        if max_freq is None:
            break
        splits = merge_pair(*best_pair, splits, word_freqs)
        merges[best_pair] = best_pair[0] + best_pair[1]
        vocab[best_pair[0] + best_pair[1]] = max_freq

    vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    for word, freq in vocab:
        dictionary.add_word(word, freq)
    
    with open(merge_dict, "w") as f:
        for pair, merge in merges.items():
            f.write(f"{pair[0]} {pair[1]} -> {merge}\n")
    return dictionary

def tokenize(text="", merge_dict=None):
    merges = {}
    # load merges
    with open(merge_dict, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pair, merge = line.split(" -> ")
            pair = tuple(pair.split())
            merges[pair] = merge
    
    words = line_split(text)
    splits = [[c for c in word] for word in words]
    for pair, merge in merges.items():
        for idx, split in enumerate(splits):
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [merge] + split[i + 2 :]
                else:
                    i += 1
            splits[idx] = split

    return sum(splits, [])

def make_binary_dataset(input_file, output_file, dictionary, append_eos=True, merge_dict=None):
    nsent, ntok = 0, 0
    unk_counter = collections.Counter()

    def unk_consumer(word, idx):
        if idx == dictionary.unk_idx and word != dictionary.unk_word:
            unk_counter.update([word])

    tokens_list = []
    with open(input_file, 'r') as inf:
        for line in inf:
            tokens = dictionary.binarize(line.strip(), tokenize, append_eos, consumer=unk_consumer, merge_dict=merge_dict)
            nsent, ntok = nsent + 1, ntok + len(tokens)
            tokens_list.append(tokens.numpy())

    with open(output_file, 'wb') as outf:
        pickle.dump(tokens_list, outf, protocol=pickle.DEFAULT_PROTOCOL)
        if not args.quiet:
            logging.info('Built a binary dataset for {}: {} sentences, {} tokens, {:.3f}% replaced by unknown token'.format(
            input_file, nsent, ntok, 100.0 * sum(unk_counter.values()) / ntok, dictionary.unk_word))


if __name__ == '__main__':
    args = get_args()
    if not args.quiet:
        utils.init_logging(args)
        logging.info('COMMAND: %s' % ' '.join(sys.argv))
        logging.info('Arguments: {}'.format(vars(args)))
    main(args)
