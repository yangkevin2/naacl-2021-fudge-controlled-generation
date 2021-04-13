import os
import random
import time
import pickle
import math
from argparse import ArgumentParser
from collections import namedtuple

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline, set_seed, GPT2Tokenizer, GPT2Model, MarianTokenizer, MarianMTModel

from data import Dataset
from model import Model
from util import save_checkpoint, ProgressMeter, AverageMeter, num_params
from constants import *
from predict_formality import predict_formality

def main(args):
    with open(args.dataset_info, 'rb') as rf:
        dataset_info = pickle.load(rf)
    tokenizer = MarianTokenizer.from_pretrained(args.model_string)
    tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})
    pad_id = tokenizer.encode(PAD_TOKEN)[0]
    model = MarianMTModel.from_pretrained(args.model_string, return_dict=True).to(args.device)
    if args.model_path is not None:
        if os.path.isdir(args.model_path):
            for _, _, files in os.walk(args.model_path):
                for fname in files:
                    if fname.endswith('.ckpt'):
                        args.model_path = os.path.join(args.model_path, fname)
                        break
        ckpt = torch.load(args.model_path)
        try:
            model.load_state_dict(ckpt['state_dict'])
        except:
            state_dict = {}
            for key in ckpt['state_dict'].keys():
                assert key.startswith('model.')
                state_dict[key[6:]] = ckpt['state_dict'][key]
            model.load_state_dict(state_dict)
    model.eval()

    checkpoint = torch.load(args.ckpt, map_location=args.device)
    model_args = checkpoint['args']
    conditioning_model = Model(model_args, pad_id, len(dataset_info.index2word)) # no need to get the glove embeddings when reloading since they're saved in model ckpt anyway
    conditioning_model.load_state_dict(checkpoint['state_dict'])
    conditioning_model = conditioning_model.to(args.device)
    conditioning_model.eval()
    if args.verbose:
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.ckpt, checkpoint['epoch']))
        print('num params', num_params(conditioning_model))

    inputs = []
    with open(args.in_file, 'r') as rf:
        for line in rf:
            inputs.append(line.strip())
    
    for inp in tqdm(inputs, total=len(inputs)):
        results = predict_formality(model, 
                        tokenizer, 
                        conditioning_model, 
                        [inp], 
                        dataset_info, 
                        precondition_topk=args.precondition_topk,
                        do_sample=args.do_sample,
                        length_cutoff=args.length_cutoff,
                        condition_lambda=args.condition_lambda,
                        device=args.device)
        print(results[0])


if __name__=='__main__':
    parser = ArgumentParser()

    # DATA
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--dataset_info', type=str, required=True, help='saved dataset info')
    parser.add_argument('--model_string', type=str, default='Helsinki-NLP/opus-mt-es-en')
    parser.add_argument('--model_path', type=str, default=None)

    parser.add_argument('--in_file', type=str, default=None, required=True, help='file containing text to run pred on')

    parser.add_argument('--precondition_topk', type=int, default=200, help='consider top k outputs from gpt at each step before conditioning and re-pruning')
    parser.add_argument('--do_sample', action='store_true', default=False, help='sample or greedy; only greedy implemented')
    parser.add_argument('--condition_lambda', type=float, default=1.0, help='lambda weight on conditioning model')
    parser.add_argument('--length_cutoff', type=int, default=512, help='max length')

    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)