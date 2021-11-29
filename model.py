import math

import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline, set_seed, GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, GPT2Config, GPT2ForSequenceClassification, GPT2LMHeadModel, MarianTokenizer

from constants import *
from util import pad_mask

class Model(nn.Module):
    def __init__(self, args, gpt_pad_id, vocab_size, rhyme_group_size=None, glove_embeddings=None, verbose=True):
        super(Model, self).__init__()

        self.topic = args.task == 'topic'
        self.formality = args.task == 'formality'
        self.iambic = args.task == 'iambic'
        self.rhyme = args.task == 'rhyme'
        self.newline = args.task == 'newline'
        if self.topic:
            self.gpt_embed = nn.Embedding(gpt_pad_id + 1, HIDDEN_DIM, padding_idx=gpt_pad_id) # these are subwords, not words
            if glove_embeddings is None:
                if verbose:
                    print('initializing word embeddings from scratch')
                self.word_embed = nn.Embedding(vocab_size, GLOVE_DIM, padding_idx=0)
            else:
                if verbose:
                    print('initializing word embeddings from glove')
                self.word_embed = nn.Embedding.from_pretrained(glove_embeddings, padding_idx=0)
            self.rnn = nn.LSTM(HIDDEN_DIM, RNN_DIM, num_layers=3, bidirectional=True)
            self.attention_linear = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
            large_hidden_dim = HIDDEN_DIM
            self.embed_key_linear = nn.Linear(large_hidden_dim, HIDDEN_DIM)
            self.attention_value_linear = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
            self.out_embed_linear = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
            self.out_linear = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
            self.out_linear2 = nn.Linear(HIDDEN_DIM + large_hidden_dim, HIDDEN_DIM)
            self.out_linear3 = nn.Linear(HIDDEN_DIM, 1)
            self.nonlinear = nn.ReLU()
        elif self.formality:
            self.marian_embed = nn.Embedding(gpt_pad_id + 1, HIDDEN_DIM, padding_idx=0) # 0 in marian is ''
            self.rnn = nn.LSTM(HIDDEN_DIM, HIDDEN_DIM, num_layers=3, bidirectional=False, dropout=0.5) # want it to be causal so we can learn all positions
            self.out_linear = nn.Linear(HIDDEN_DIM, 1)
        elif self.iambic:
            self.gpt_embed = nn.Embedding(gpt_pad_id + 1, HIDDEN_DIM, padding_idx=gpt_pad_id)
            self.rnn = nn.LSTM(HIDDEN_DIM, HIDDEN_DIM, num_layers=3, bidirectional=False, dropout=0) # want it to be causal so we can learn all positions
            self.out_linear = nn.Linear(HIDDEN_DIM, 1)
        elif self.rhyme:
            self.gpt_embed = nn.Embedding(gpt_pad_id + 1, HIDDEN_DIM, padding_idx=gpt_pad_id) # these are subwords, not words
            self.word_embed = nn.Embedding(rhyme_group_size+1, GLOVE_DIM, padding_idx=0) # this embedding for future words will actually embed the rhyme group idx
            self.rnn = nn.LSTM(HIDDEN_DIM, RNN_DIM, num_layers=3, bidirectional=True)
            self.attention_linear = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
            large_hidden_dim = HIDDEN_DIM + COUNT_SYLLABLE_DIM
            self.embed_key_linear = nn.Linear(large_hidden_dim, HIDDEN_DIM)
            self.attention_value_linear = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
            self.out_embed_linear = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
            self.out_linear = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
            self.out_linear2 = nn.Linear(HIDDEN_DIM + large_hidden_dim, HIDDEN_DIM)
            self.out_linear3 = nn.Linear(HIDDEN_DIM, 1)
            self.count_syllable_embed = nn.Embedding(MAX_COUNT_SYLLABLE_DIST+1, COUNT_SYLLABLE_DIM)
            self.nonlinear = nn.ReLU()
        elif self.newline:
            self.gpt_embed = nn.Embedding(gpt_pad_id + 1, HIDDEN_DIM, padding_idx=gpt_pad_id) # these are subwords, not words
            self.rnn = nn.LSTM(HIDDEN_DIM, HIDDEN_DIM, num_layers=3, bidirectional=False)
            self.count_syllable_embed = nn.Embedding(MAX_COUNT_SYLLABLE_DIST+1, COUNT_SYLLABLE_DIM)
            self.out_linear = nn.Linear(HIDDEN_DIM + COUNT_SYLLABLE_DIM, HIDDEN_DIM)
            self.out_linear2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
            self.out_linear3 = nn.Linear(HIDDEN_DIM, 1)
            self.nonlinear = nn.ReLU()
        else:
            raise NotImplementedError # TODO honestly this can/should be refactored into different models


    def forward(self, inputs, lengths=None, future_words=None, log_probs=None, syllables_to_go=None, future_word_num_syllables=None, rhyme_group_index=None, run_classifier=False):
        """
        inputs: token ids, batch x seq, right-padded with 0s
        lengths: lengths of inputs; batch
        future_words: batch x N words to check if not predict next token, else batch
        log_probs: N
        syllables_to_go: batch
        """
        if self.topic:
            inputs = self.gpt_embed(inputs) # batch x seq x 300
            inputs = pack_padded_sequence(inputs.permute(1, 0, 2), lengths.cpu(), enforce_sorted=False)
            rnn_output, _ = self.rnn(inputs)
            rnn_output, _ = pad_packed_sequence(rnn_output)
            rnn_output = rnn_output.permute(1, 0, 2) # batch x seq x 300
            hidden = rnn_output
            attention_mask = pad_mask(lengths).permute(1, 0) # batch x seq
            embed = self.word_embed(future_words) # batch x N x 300
            embed_query = self.embed_key_linear(embed)
            attention_tensor = self.attention_linear(hidden).unsqueeze(2) * embed_query.unsqueeze(1) # batch x seq x N x 300
            attention_weights = F.softmax(attention_tensor.sum(dim=3), dim=1) # batch x seq x N
            attention_weights = attention_weights * attention_mask.unsqueeze(2)
            hidden = self.attention_value_linear(hidden)
            weighted_hidden = (hidden.unsqueeze(2) * attention_weights.unsqueeze(3)).sum(dim=1) # batch x seq x N x 768 -> batch x N x 768
            unnormalized_scores = (self.out_linear(weighted_hidden) * self.out_embed_linear(embed)) # batch x N x 300
            unnormalized_scores = torch.cat([unnormalized_scores, embed], dim=2)
            unnormalized_scores = self.nonlinear(self.out_linear2(self.nonlinear(unnormalized_scores)))
            unnormalized_scores = self.out_linear3(unnormalized_scores)
            scores = unnormalized_scores.squeeze(2) - log_probs.unsqueeze(0) 
            return scores # batch x N of normalized scores or batch x 
        elif self.formality:
            inputs = self.marian_embed(inputs)
            inputs = pack_padded_sequence(inputs.permute(1, 0, 2), lengths.cpu(), enforce_sorted=False)
            rnn_output, _ = self.rnn(inputs)
            rnn_output, _ = pad_packed_sequence(rnn_output)
            rnn_output = rnn_output.permute(1, 0, 2) # batch x seq x 300
            return self.out_linear(rnn_output).squeeze(2)
        elif self.iambic:
            inputs = self.gpt_embed(inputs)
            inputs = pack_padded_sequence(inputs.permute(1, 0, 2), lengths.cpu(), enforce_sorted=False)
            rnn_output, _ = self.rnn(inputs)
            rnn_output, _ = pad_packed_sequence(rnn_output)
            rnn_output = rnn_output.permute(1, 0, 2) # batch x seq x 300
            return self.out_linear(rnn_output).squeeze(2)
        elif self.rhyme:
            inputs = self.gpt_embed(inputs) # batch x seq x 300
            inputs = pack_padded_sequence(inputs.permute(1, 0, 2), lengths.cpu(), enforce_sorted=False)
            rnn_output, _ = self.rnn(inputs)
            rnn_output, _ = pad_packed_sequence(rnn_output)
            rnn_output = rnn_output.permute(1, 0, 2) # batch x seq x 300
            hidden = rnn_output
            attention_mask = pad_mask(lengths).permute(1, 0) # batch x seq
            embed = self.word_embed(future_words) # batch x N x 300
            embedded_syllables_to_go = self.count_syllable_embed(syllables_to_go).unsqueeze(1).expand(-1, embed.shape[1], -1) # batch x N x 100
            auxiliary_embed = embedded_syllables_to_go
            embed_query = self.embed_key_linear(torch.cat([embed, auxiliary_embed], dim=2))
            attention_tensor = self.attention_linear(hidden).unsqueeze(2) * embed_query.unsqueeze(1) # batch x seq x N x 300
            attention_weights = F.softmax(attention_tensor.sum(dim=3), dim=1) # batch x seq x N
            attention_weights = attention_weights * attention_mask.unsqueeze(2)
            hidden = self.attention_value_linear(hidden)
            weighted_hidden = (hidden.unsqueeze(2) * attention_weights.unsqueeze(3)).sum(dim=1) # batch x seq x N x 768 -> batch x N x 768
            unnormalized_scores = (self.out_linear(weighted_hidden) * self.out_embed_linear(embed)) # batch x N x 300
            unnormalized_scores = torch.cat([unnormalized_scores, embed, auxiliary_embed], dim=2)
            unnormalized_scores = self.nonlinear(self.out_linear2(self.nonlinear(unnormalized_scores)))
            unnormalized_scores = self.out_linear3(unnormalized_scores)
            scores = unnormalized_scores.squeeze(2) - log_probs.unsqueeze(0) 
            return scores # batch x N of normalized scores or batch x 
        elif self.newline:
            inputs = self.gpt_embed(inputs) # batch x seq x 300
            inputs = pack_padded_sequence(inputs.permute(1, 0, 2), lengths.cpu(), enforce_sorted=False)
            rnn_output, _ = self.rnn(inputs)
            rnn_output, _ = pad_packed_sequence(rnn_output)
            rnn_output = rnn_output.permute(1, 0, 2) # batch x seq x 300
            hidden = torch.cat([rnn_output, self.count_syllable_embed(syllables_to_go).unsqueeze(1).expand(-1, rnn_output.shape[1], -1)], dim=2)
            return self.out_linear3(self.nonlinear(self.out_linear2(self.nonlinear(self.out_linear(hidden))))).squeeze(2)
        else: 
            raise NotImplementedError
            
