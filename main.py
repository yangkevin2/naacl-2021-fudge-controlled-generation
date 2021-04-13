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

from data import Dataset
from model import Model
from util import save_checkpoint, ProgressMeter, AverageMeter, num_params, pad_mask
from constants import *


def train(model, dataset, optimizer, criterion, epoch, args, data_start_index):
    model.train()
    if data_start_index == 0:
        dataset.shuffle('train', seed=epoch + args.seed)
    if args.epoch_max_len is not None:
        data_end_index = min(data_start_index + args.epoch_max_len, len(dataset.splits['train']))
        loader = dataset.loader('train', num_workers=args.num_workers, indices=list(range(data_start_index, data_end_index)))
        data_start_index = data_end_index if data_end_index < len(dataset.splits['train']) else 0
    else:
        loader = dataset.loader('train', num_workers=args.num_workers)
    loss_meter = AverageMeter('loss', ':6.4f')
    total_length = len(loader)
    progress = ProgressMeter(total_length, [loss_meter], prefix='Training: ')
    for batch_num, batch in enumerate(tqdm(loader, total=len(loader))):
        batch = [tensor.to(args.device) for tensor in batch]
        inputs, lengths, future_words, log_probs, labels, classification_targets, syllables_to_go, future_word_num_syllables, rhyme_group_index = batch
        if args.task not in ['formality', 'iambic']:
            if not args.debug and len(inputs) != args.batch_size: # it'll screw up the bias...?
                continue
        scores = model(inputs, lengths, future_words, log_probs, syllables_to_go, future_word_num_syllables, rhyme_group_index, run_classifier=True)
        if args.task == 'formality': # we're learning for all positions at once. scores are batch x seq
            expanded_labels = classification_targets.unsqueeze(1).expand(-1, scores.shape[1]) # batch x seq
            length_mask = pad_mask(lengths).permute(1, 0) # batch x seq
            loss = criterion(scores.flatten()[length_mask.flatten()==1], expanded_labels.flatten().float()[length_mask.flatten()==1])
        elif args.task in ['iambic', 'newline']:
            use_indices = classification_targets.flatten() != -1
            loss = criterion(scores.flatten()[use_indices], classification_targets.flatten().float()[use_indices])
        else: # topic, rhyme
            loss = criterion(scores.flatten(), labels.flatten().float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.detach(), len(labels))
        if batch_num % args.train_print_freq == 0:
            progress.display(batch_num)
    progress.display(total_length)
    return data_start_index


def validate(model, dataset, criterion, epoch, args):
    model.eval()
    random.seed(0)
    loader = dataset.loader('val', num_workers=args.num_workers)
    loss_meter = AverageMeter('loss', ':6.4f')
    total_length = len(loader)
    progress = ProgressMeter(total_length, [loss_meter], prefix='Validation: ')
    with torch.no_grad():
        for batch_num, batch in enumerate(tqdm(loader, total=len(loader))):
            batch = [tensor.to(args.device) for tensor in batch]
            inputs, lengths, future_words, log_probs, labels, classification_targets, syllables_to_go, future_word_num_syllables, rhyme_group_index = batch
            if args.task not in ['formality', 'iambic']: # topic predictor
                if not args.debug and len(inputs) != args.batch_size:
                    continue
            scores = model(inputs, lengths, future_words, log_probs, syllables_to_go, future_word_num_syllables, rhyme_group_index, run_classifier=True)
            if args.task == 'formality': # we're learning for all positions at once. scores are batch x seq
                expanded_labels = classification_targets.unsqueeze(1).expand(-1, scores.shape[1]) # batch x seq
                length_mask = pad_mask(lengths).permute(1, 0) # batch x seq
                loss = criterion(scores.flatten()[length_mask.flatten()==1], expanded_labels.flatten().float()[length_mask.flatten()==1])
            elif args.task in ['iambic', 'newline']:
                use_indices = classification_targets.flatten() != -1
                loss = criterion(scores.flatten()[use_indices], classification_targets.flatten().float()[use_indices])
            else: # topic, rhyme
                loss = criterion(scores.flatten(), labels.flatten().float())
            loss_meter.update(loss.detach(), len(labels))
            if batch_num % args.train_print_freq == 0:
                progress.display(batch_num)
    progress.display(total_length)
    return loss_meter.avg


def main(args):
    dataset = Dataset(args)
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'dataset_info'), 'wb') as wf:
        pickle.dump(dataset.dataset_info, wf)
    if args.task == 'rhyme':
        with open(os.path.join(args.save_dir, 'rhyme_info'), 'wb') as wf:
            pickle.dump(dataset.rhyme_info, wf)
    if args.ckpt:
        checkpoint = torch.load(args.ckpt, map_location=args.device)
        start_epoch = checkpoint['epoch'] + 1
        best_val_metric = checkpoint['best_metric']
        model_args = checkpoint['args']
        model = Model(model_args, dataset.gpt_pad_id, len(dataset.index2word), rhyme_group_size=len(dataset.index2rhyme_group) if args.task == 'rhyme' else None) # no need to get the glove embeddings when reloading since they're saved in model ckpt anyway
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=model_args.lr)
        optimizer.load_state_dict(checkpoint['optimizer'])
        data_start_index = checkpoint['data_start_index']
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.ckpt, checkpoint['epoch']))
        # NOTE: just import pdb after loading the model here if you want to play with it, it's easy
        # model.eval()
        # import pdb; pdb.set_trace()
    else:
        model = Model(args, dataset.gpt_pad_id, len(dataset.index2word), rhyme_group_size=len(dataset.index2rhyme_group) if args.task == 'rhyme' else None, glove_embeddings=dataset.glove_embeddings)
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_val_metric = 1e8 # lower is better for BCE
        data_start_index = 0
    print('num params', num_params(model))
    criterion = nn.BCEWithLogitsLoss().to(args.device)
    
    if args.evaluate:
        epoch = 0
        validate(model, dataset, criterion, epoch, args)
        return
    for epoch in range(args.epochs):
        print("TRAINING: Epoch {} at {}".format(epoch, time.ctime()))
        data_start_index = train(model, dataset, optimizer, criterion, epoch, args, data_start_index)
        if epoch % args.validation_freq == 0:
            print("VALIDATION: Epoch {} at {}".format(epoch, time.ctime()))
            metric = validate(model, dataset, criterion, epoch, args)

            if not args.debug:
                if metric < best_val_metric:
                    print('new best val metric', metric)
                    best_val_metric = metric
                    save_checkpoint({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'best_metric': best_val_metric,
                        'optimizer': optimizer.state_dict(),
                        'data_start_index': data_start_index,
                        'args': args
                    }, os.path.join(args.save_dir, 'model_best.pth.tar'))
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_metric': metric,
                    'optimizer': optimizer.state_dict(),
                    'data_start_index': data_start_index,
                    'args': args
                }, os.path.join(args.save_dir, 'model_epoch' + str(epoch) + '.pth.tar'))


if __name__=='__main__':
    parser = ArgumentParser()

    # DATA
    parser.add_argument('--task', type=str, required=True, choices=['iambic', 'rhyme', 'newline', 'topic', 'formality'])
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--glove_file', type=str, help='glove embedding init, for topic task')

    # SAVE/LOAD
    parser.add_argument('--save_dir', type=str, required=True, help='where to save ckpts')
    parser.add_argument('--ckpt', type=str, default=None, help='load ckpt from file if given')
    parser.add_argument('--dataset_info', type=str, help='saved dataset info')
    parser.add_argument('--rhyme_info', type=str, help='saved dataset rhyme info, for a ckpt with task==rhyme')

    # TRAINING
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--epoch_max_len', type=int, default=None, help='max batches per epoch if set, for more frequent validation')
    parser.add_argument('--validation_freq', type=int, default=1, help='validate every X epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Adam learning rate')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--num_workers', type=int, default=20, help='num workers for data loader')
    parser.add_argument('--evaluate', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)

    # PRINTING
    parser.add_argument('--train_print_freq', type=int, default=100, help='how often to print metrics (every X batches)')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.evaluate:
        assert args.ckpt is not None
    
    main(args)