from argparse import ArgumentParser
import math
import string

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForSequenceClassification

from poetry_util import is_iambic, perfect_rhyme_end, count_syllables
from constants import *


def conditional_perplexity(prefix, pred, tokenizer, model, device='cuda', sep_losses=False):
    # calculate perplexity on pred only, conditioned on prefix
    sentence = prefix + pred
    sos_token = tokenizer.decode([0])
    prefix_tensor_input = tokenizer.encode(sos_token + prefix.replace(EOT_TOKEN, ' ').strip(), return_tensors='pt').to(device)
    full_tensor_input = tokenizer.encode(sos_token + sentence.replace(EOT_TOKEN, ' ').strip(), return_tensors='pt').to(device)
    if sep_losses:
        prefix_loss = model(prefix_tensor_input, labels=prefix_tensor_input)[0].sum()
        full_loss = model(full_tensor_input, labels=full_tensor_input)[0].sum()
    else:
        prefix_loss = model(prefix_tensor_input, labels=prefix_tensor_input)[0] * (prefix_tensor_input.shape[1]-1) # neg log prob of prefix
        full_loss = model(full_tensor_input, labels=full_tensor_input)[0] * (full_tensor_input.shape[1]-1) # neg log prob of full seq
    pred_loss = full_loss - prefix_loss # neg log prob of preds given prefix
    avg_pred_loss = pred_loss / (full_tensor_input.shape[1] - prefix_tensor_input.shape[1])
    return math.exp(avg_pred_loss.item())


def grammaticality(sentences, tokenizer, model, device='cuda'):
    with torch.no_grad():
        total_good = 0
        for sent in tqdm(sentences, total=len(sentences)):
            good_prob = F.softmax(model(tokenizer.encode(sent, return_tensors='pt').to(device))[0].flatten(), dim=0)[1]
            total_good += good_prob
        return total_good / len(sentences) # avg probability of grammaticality according to model


def distinctness(sentences):
    d1 = set()
    d2 = set()
    d3 = set()
    total_words = 0
    for sentence in sentences:
        o = sentence.split(' ')
        total_words += len(o)
        d1.update(o)
        for i in range(len(o) - 1):
            d2.add(o[i] + '_' + o[i+1])
        for i in range(len(o) - 2):
            d3.add(o[i] + '_' + o[i+1] + '_' + o[i+2])
    return len(d1) / total_words, len(d2) / total_words, len(d3) / total_words


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--pred_file', type=str)
    parser.add_argument('--prefix_file', type=str)
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    args = parser.parse_args()

    preds = []
    with open(args.pred_file, 'r') as rf:
        for line in rf:
            preds.append(line[:-1]) # drop \n but not beginning spaces if any
    prefixes = []
    with open(args.prefix_file, 'r') as rf:
        for line in rf:
            prefixes.append(line.strip())
    assert len(prefixes) == len(preds)
    rhymes = 0
    iambic = 0
    ten_syllables = 0
    end = 0
    diff_rhymes = 0
    all_success = 0
    total = len(prefixes)
    for prefix, pred in zip(prefixes, preds):
        if is_iambic(pred):
            iambic += 1
        if perfect_rhyme_end(prefix, pred):
            rhymes += 1
            if prefix.split()[-1].strip(string.punctuation) != pred.split()[-1].strip(string.punctuation):
                diff_rhymes += 1
        if count_syllables(pred) == 10:
            ten_syllables += 1
        if pred.strip()[-1] in PHRASE_ENDS:
            end += 1
        if is_iambic(pred) and perfect_rhyme_end(prefix, pred) and count_syllables(pred) == 10 and pred.strip()[-1] in PHRASE_ENDS:
            all_success += 1
    print('iambic', iambic, 'out of', total, ', frac', iambic / total)
    print('rhymes', rhymes, 'out of', total, ', frac', rhymes / total)
    print('end sentence', end, 'out of', total, ', frac', end / total)
    print('10 syllables', ten_syllables, 'out of', total, ', frac', ten_syllables / total)
    print('all success', all_success, 'out of', total, ', frac', all_success / total)
    print('rhymes with diff word', diff_rhymes, 'out of', total, ', frac', diff_rhymes / total)

    print('distinctness', distinctness(preds))

    grammar_tokenizer = AutoTokenizer.from_pretrained('textattack/roberta-base-CoLA')
    grammar_model = AutoModelForSequenceClassification.from_pretrained('textattack/roberta-base-CoLA').to(args.device)
    grammar_model.eval()
    print('grammaticality', grammaticality(preds, grammar_tokenizer, grammar_model, device=args.device))

    perplexities = []
    eval_tokenizer = AutoTokenizer.from_pretrained('transfo-xl-wt103')
    eval_model = AutoModelWithLMHead.from_pretrained('transfo-xl-wt103').to(args.device)
    eval_model.eval()
    for prefix, pred in zip(prefixes, preds):
        perplexities.append(conditional_perplexity(prefix, pred, eval_tokenizer, eval_model, device=args.device, sep_losses=True))
    print('transformer xl perplexity', np.mean(perplexities), '+/-', np.std(perplexities))

    perplexities = []
    eval_tokenizer = AutoTokenizer.from_pretrained('openai-gpt')
    eval_model = AutoModelWithLMHead.from_pretrained('openai-gpt').to(args.device)
    eval_model.eval()
    for prefix, pred in zip(prefixes, preds):
        perplexities.append(conditional_perplexity(prefix, pred, eval_tokenizer, eval_model, device=args.device))
    print('gpt perplexity', np.mean(perplexities), '+/-', np.std(perplexities))

    # NOTE: uncomment this section with the path to the Shakespeare-finetuned GPT to evaluate this metric. it's in ckpt/poetry/gpt_finetune_shakespeare.pth.tar. 
    # eval_tokenizer = AutoTokenizer.from_pretrained('openai-gpt')
    # eval_model = AutoModelWithLMHead.from_pretrained('openai-gpt').to(args.device)
    # checkpoint = torch.load('***PATH_TO_SHAKESPEARE_FINETUNED_GPT***', map_location=args.device)
    # mod_dict = {}
    # for key in checkpoint['state_dict']:
    #     mod_dict[key.replace('classifier.', '')] = checkpoint['state_dict'][key]
    # eval_model.load_state_dict(mod_dict)
    # eval_model.eval()
    # perplexities = []
    # for prefix, pred in zip(prefixes, preds):
    #     perplexities.append(conditional_perplexity(prefix, pred, eval_tokenizer, eval_model, device=args.device))
    # print('shakespeare finetuned perplexity', np.mean(perplexities), '+/-', np.std(perplexities))
