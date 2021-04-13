from argparse import ArgumentParser
import pickle
import os
import math

import sacrebleu
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline, set_seed, GPT2Tokenizer, GPT2Model, MarianTokenizer, MarianMTModel

from constants import *
from model import Model
from util import save_checkpoint, ProgressMeter, AverageMeter, num_params

def avg_formality(preds, model, tokenizer, device='cuda'):
    probs = []
    for sent in preds:
        encoded_input = tokenizer.encode(sent, return_tensors='pt').to(device)
        lengths = torch.LongTensor([encoded_input.shape[1]]).to(device)
        scores = model(encoded_input, lengths=lengths) # batch x seq
        score = scores.flatten()[-1].item()
        probs.append(math.exp(score) / (1 + math.exp(score))) # sigmoided score = prob
    return np.mean(probs)

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--pred', type=str)
    parser.add_argument('--ref', type=str, nargs='*', help='bleu refs')
    parser.add_argument('--ckpt', type=str, help='formality classifier')
    parser.add_argument('--dataset_info', type=str)
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--model_string', type=str, default='Helsinki-NLP/opus-mt-es-en')

    args = parser.parse_args()

    # refs = [['The dog bit the man.', 'It was not unexpected.', 'The man bit him first.'],
    #         ['The dog had bit the man.', 'No one was surprised.', 'The man had bitten the dog.']]
    # sys = ['The dog bit the man.', "It wasn't surprising.", 'The man had just bitten him.']
    print('num ref files', len(args.ref))
    pred = []
    with open(args.pred, 'r') as rf:
        for line in rf:
            pred.append(line.strip())
    refs = []
    for ref_file in args.ref:
        ref = []
        with open(ref_file, 'r') as rf:
            for line in rf:
                ref.append(line.strip())
        assert len(ref) == len(pred)
        refs.append(ref)
    bleu = sacrebleu.corpus_bleu(pred, refs)
    print('BLEU score:', bleu.score)

    with open(args.dataset_info, 'rb') as rf:
        dataset_info = pickle.load(rf)

    tokenizer = MarianTokenizer.from_pretrained(args.model_string)
    tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})
    pad_id = tokenizer.encode(PAD_TOKEN)[0]

    checkpoint = torch.load(args.ckpt, map_location=args.device)
    model_args = checkpoint['args']
    conditioning_model = Model(model_args, pad_id, len(dataset_info.index2word)) # no need to get the glove embeddings when reloading since they're saved in model ckpt anyway
    conditioning_model.load_state_dict(checkpoint['state_dict'])
    conditioning_model = conditioning_model.to(args.device)
    conditioning_model.eval()
    print("=> loaded checkpoint '{}' (epoch {})"
            .format(args.ckpt, checkpoint['epoch']))
    print('num params', num_params(conditioning_model))

    print('avg formality prob according to model', avg_formality(pred, conditioning_model, tokenizer, device=args.device))

