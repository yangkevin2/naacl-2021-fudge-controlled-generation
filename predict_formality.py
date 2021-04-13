import os
import random
import time
import pickle
import math
from argparse import ArgumentParser

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

def main(args):
    with open(args.dataset_info, 'rb') as rf:
        dataset_info = pickle.load(rf)
    tokenizer = MarianTokenizer.from_pretrained(args.model_string)
    tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})
    pad_id = tokenizer.encode(PAD_TOKEN)[0]
    model = MarianMTModel.from_pretrained(args.model_string, return_dict=True).to(args.device)
    model.eval()

    checkpoint = torch.load(args.ckpt, map_location=args.device)
    model_args = checkpoint['args']
    conditioning_model = Model(model_args, pad_id, len(dataset_info.index2word)) # no need to get the glove embeddings when reloading since they're saved in model ckpt anyway
    conditioning_model.load_state_dict(checkpoint['state_dict'])
    conditioning_model = conditioning_model.to(args.device)
    conditioning_model.eval()
    print("=> loaded checkpoint '{}' (epoch {})"
            .format(args.ckpt, checkpoint['epoch']))
    print('num params', num_params(conditioning_model))

    while True:
        results = predict_formality(model, 
                        tokenizer, 
                        conditioning_model, 
                        [args.input_text], 
                        dataset_info, 
                        precondition_topk=args.precondition_topk,
                        do_sample=args.do_sample,
                        length_cutoff=args.length_cutoff,
                        condition_lambda=args.condition_lambda,
                        device=args.device)
        print(results)
        import pdb; pdb.set_trace()


def predict_formality(model, tokenizer, conditioning_model, input_text, dataset_info, precondition_topk=200, do_sample=False, length_cutoff=512, condition_lambda=1.0, device='cuda'):
    with torch.no_grad():
        batch_size = len(input_text)

        # assumes initially all same length.
        encoded_input = [tokenizer.encode(it, return_tensors='pt').to(device) for it in input_text] # batch x seq
        encoded_input = torch.cat(encoded_input, dim=0)

        input_ids = torch.LongTensor([[65000]]).to(device)
        cur_len = 1
        max_length = length_cutoff
        min_length = 0
        temperature = 1.0
        top_k = 50
        top_p = 1.0
        repetition_penalty = 1.0
        no_repeat_ngram_size = 0
        bad_words_ids = [[65000]]
        pad_token_id = 65000
        eos_token_id = 0
        effective_batch_size = batch_size
        attention_mask = encoded_input.new_ones(encoded_input.shape)
        use_cache = True
        model_specific_kwargs = {'encoder_outputs': model.get_encoder()(encoded_input, attention_mask=attention_mask)}

        output = _generate_no_beam_search(model,
                                        conditioning_model,
                                        condition_lambda,
                                        precondition_topk,
                                        input_ids,
                                        cur_len,
                                        max_length,
                                        min_length,
                                        do_sample,
                                        temperature,
                                        top_k,
                                        top_p,
                                        repetition_penalty,
                                        no_repeat_ngram_size,
                                        bad_words_ids,
                                        pad_token_id,
                                        eos_token_id,
                                        batch_size,
                                        attention_mask,
                                        use_cache,
                                        model_specific_kwargs)

        return [tokenizer.decode(s[1:]) for s in output] # 1: to delete the pad token


# hack of code from transformers/generation_utils.py
# to get our conditioning
def _generate_no_beam_search(
        model,
        conditioning_model,
        condition_lambda,
        precondition_topk,
        input_ids,
        cur_len,
        max_length,
        min_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        pad_token_id,
        eos_token_id,
        batch_size,
        attention_mask,
        use_cache,
        model_kwargs,
    ):
        """Generate sequences for each example without beam search (num_beams == 1).
        All returned sequence are generated independantly.
        """
        # length of generated sentences / unfinished sentences
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        sent_lengths = input_ids.new(batch_size).fill_(max_length)

        past = None
        while cur_len < max_length:
            model_inputs = model.prepare_inputs_for_generation(
                input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_kwargs
            )

            outputs = model(**model_inputs, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]

            scores = model.postprocess_next_token_scores(
                scores=next_token_logits,
                input_ids=input_ids,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                cur_len=cur_len,
                min_length=min_length,
                max_length=max_length,
                eos_token_id=eos_token_id,
                repetition_penalty=repetition_penalty,
                batch_size=batch_size,
                num_beams=1,
            )

            # if model has past, then set the past variable to speed up decoding
            if "past_key_values" in outputs:
                past = outputs.past_key_values
            elif "mems" in outputs:
                past = outputs.mems

            top_logits, top_indices = scores.topk(precondition_topk, dim=1) # batch x topk
            tplus1_candidates = torch.cat([input_ids.unsqueeze(1).expand(-1, precondition_topk, -1), top_indices.unsqueeze(2)], dim=2)[:, :, 1:] # batch x topk x seq+1, with pad dropped
            expanded_lengths = torch.LongTensor([[cur_len for _ in range(precondition_topk)] for _ in range(batch_size)]).to(scores.device)
            if condition_lambda == 0:
                condition_logits = torch.zeros_like(top_logits).float()
            else:
                condition_logits = conditioning_model(tplus1_candidates.flatten(0, 1), # batch*topk x seq+1
                                                    expanded_lengths.flatten(0, 1), # batch*topk
                                                    None,
                                                    None,
                                                    None)
                condition_logits = condition_logits.view(batch_size, precondition_topk, -1)[:, :, -1] # batch x topk of last formality pred
                condition_logits = condition_logits - torch.log(1 + torch.exp(condition_logits)) # get correct log probs
                # condition_logits = - torch.log(1 + torch.exp(condition_logits)) # for informal
            full_logits = top_logits + condition_lambda * condition_logits
            if do_sample:
                raise NotImplementedError
            else:
                # Greedy decoding
                next_token = top_indices[torch.arange(batch_size).to(top_indices.device), torch.argmax(full_logits, dim=-1)]

            # if do_sample:
            #     # Temperature (higher temperature => more likely to sample low probability tokens)
            #     if temperature != 1.0:
            #         scores = scores / temperature
            #     # Top-p/top-k filtering
            #     next_token_logscores = top_k_top_p_filtering(scores, top_k=top_k, top_p=top_p)
            #     # Sample
            #     probs = F.softmax(next_token_logscores, dim=-1)
            #     next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            # else:
            #     # Greedy decoding
            #     next_token = torch.argmax(next_token_logits, dim=-1)

            # update generations and finished sentences
            if eos_token_id is not None:
                # pad finished sentences if eos_token_id exist
                tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
            else:
                tokens_to_add = next_token
            
            # add token and increase length by one
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            if eos_token_id is not None:
                eos_in_sents = tokens_to_add == eos_token_id
                # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
                is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
                # unfinished_sents is set to zero if eos in sentence
                unfinished_sents.mul_((~eos_in_sents).long())

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

            # extend attention_mask for new generated input if only decoder
            if model.config.is_encoder_decoder is False:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        return input_ids

if __name__=='__main__':
    parser = ArgumentParser()

    # DATA
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--dataset_info', type=str, required=True, help='saved dataset info')
    parser.add_argument('--model_string', type=str, default='Helsinki-NLP/opus-mt-es-en')

    parser.add_argument('--input_text', type=str, default=None, required=True, help='text to run pred on')

    parser.add_argument('--precondition_topk', type=int, default=200, help='consider top k outputs from gpt at each step before conditioning and re-pruning')
    parser.add_argument('--do_sample', action='store_true', default=False, help='sample instead of greedy')
    parser.add_argument('--condition_lambda', type=float, default=1.0, help='lambda weight on conditioning model')
    parser.add_argument('--length_cutoff', type=int, default=512, help='max length')

    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--debug', action='store_true', default=False)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)