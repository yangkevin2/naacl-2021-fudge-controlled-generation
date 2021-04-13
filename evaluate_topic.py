import os
import random
import time
import pickle
import math
from argparse import ArgumentParser
from collections import defaultdict
import string
import csv

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline, set_seed, GPT2Tokenizer, GPT2Model

from data import Dataset
from model import Model
from util import save_checkpoint, ProgressMeter, AverageMeter, num_params, pad_mask
from predict_topic import predict
from constants import *


def main(args):
    with open(args.dataset_info, 'rb') as rf:
        dataset_info = pickle.load(rf)
    gpt_tokenizer = AutoTokenizer.from_pretrained(args.model_string)
    gpt_tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})
    gpt_pad_id = gpt_tokenizer.encode(PAD_TOKEN)[0]
    gpt_model = AutoModelWithLMHead.from_pretrained(args.model_string).to(args.device)
    gpt_model.eval()

    checkpoint = torch.load(args.ckpt, map_location=args.device)
    model_args = checkpoint['args']
    conditioning_model = Model(model_args, gpt_pad_id, len(dataset_info.index2word)) # no need to get the glove embeddings when reloading since they're saved in model ckpt anyway
    conditioning_model.load_state_dict(checkpoint['state_dict'])
    conditioning_model = conditioning_model.to(args.device)
    conditioning_model.eval()
    if args.verbose:
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.ckpt, checkpoint['epoch']))
        print('num params', num_params(conditioning_model))

    input_texts, conditions, categories = [], [], []

    if args.condition_file is not None:
        with open(args.condition_file, 'r') as rf:
            for line in rf:
                input_texts.append(line.strip().split('\t')[0])
                conditions.append(line.strip().split('\t')[1])
                categories.append(None)
                for cw in conditions[-1].split():
                    assert cw in dataset_info.word2index
    else:
        prefixes = []
        with open(args.prefix_file, 'r') as rf:
            for line in rf:
                prefixes.append(line.strip())
        condition_wordlists = []
        for root, _, files in os.walk(args.wordlist_dir):
            for fname in files:
                words = []
                with open(os.path.join(root, fname), 'r') as rf:
                    for line in rf:
                        word = line.strip()
                        if word in dataset_info.word2index:
                            words.append(word)
                        else:
                            if args.verbose:
                                print('word not found:', word)
                condition_wordlists.append((' '.join(words), fname.split('.')[0]))
        for p in prefixes:
            for c, category in condition_wordlists:
                input_texts.append(p)
                conditions.append(c)
                categories.append(category)
    
    all_cr = []
    pair_num = 0
    for input_text, condition_words, category in tqdm(zip(input_texts, conditions, categories), total=len(conditions)):
        predict_function = predict
        condition_results = []
        for i in range(0, args.sample_size, args.max_sample_batch):
            num_samples = min(args.max_sample_batch, args.sample_size - i)
            condition_results += predict_function(gpt_model, 
                            gpt_tokenizer, 
                            conditioning_model, 
                            [input_text for _ in range(num_samples)],
                            condition_words,
                            dataset_info, 
                            args.precondition_topk,
                            args.topk, 
                            args.length_cutoff,
                            condition_lambda=args.condition_lambda,
                            device=args.device)
        all_cr.append((input_text, category, condition_results))
        pair_num += 1
        if args.max_pairs > 0 and pair_num >= args.max_pairs:
            break
    with open(args.log_file, 'w') as wf:
        writer = csv.DictWriter(wf, fieldnames=['category', 'input_text', 'generation'])
        writer.writeheader()
        for cr_group in all_cr:
            for cr in cr_group[2]:
                writer.writerow({'category': cr_group[1], 'input_text': cr_group[0], 'generation': cr})


if __name__=='__main__':
    parser = ArgumentParser()

    # DATA
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--log_file', type=str, required=True, help='file to write outputs to (csv format)')
    parser.add_argument('--dataset_info', type=str, required=True, help='saved dataset info')
    parser.add_argument('--model_string', type=str, default='gpt2-medium')

    parser.add_argument('--condition_file', type=str, default=None, help='file of inputs and conditions')
    parser.add_argument('--prefix_file', type=str, default=None, help='prefix set')
    parser.add_argument('--wordlist_dir', type=str, default=None, help='dir of bow wordlists for categories')
    parser.add_argument('--sample_size', type=int, default=3, help='samples per input text-condition pair')
    parser.add_argument('--max_sample_batch', type=int, default=3, help='max samples at a time')
    parser.add_argument('--max_pairs', type=int, default=-1, help='max input-condition pairs, for debugging quickly')

    parser.add_argument('--precondition_topk', type=int, default=200, help='consider top k outputs from gpt at each step before conditioning and re-pruning')
    parser.add_argument('--topk', type=int, default=10, help='consider top k outputs from gpt at each step')
    parser.add_argument('--condition_lambda', type=float, default=1.0, help='lambda weight on conditioning model')
    parser.add_argument('--length_cutoff', type=int, default=80, help='max length')

    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)

    args = parser.parse_args()

    assert (args.condition_file is not None) != (args.prefix_file is not None and args.wordlist_dir is not None) # one of two interfaces for specifying

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)