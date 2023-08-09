"""
GeNAS
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import math
from config import config
assert torch.cuda.is_available()

def check_strictly_increasing(L):
    return all(x<y for x, y in zip(L, L[1:]))

def no_grad_wrapper(func):
    def new_func(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return new_func

@no_grad_wrapper
def get_cand_wlm(model, genotype, train_dataloader, val_dataloader, stds=[0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05], max_train_img_size=850, max_val_img_size=25000):
    '''
        genotype: normal (14 edges) + reduce cell (14 edges) with operation indices. e.g. 6, 6, 3, 0, 4, 7, 0, 0, 1, 0, 6, 0, 6, 0, 1, 4, 3, 7, 0, 7, 0, 0, 4, 0, 0, 0, 3, 2]
        train_dataloader: half (25K) of original training set (50K).
        val_dataloader: another half (25K) of original training set (50K).
    '''
    # separate genotype
    normal_genotype = tuple(genotype[:config.edges])
    reduce_genotype = tuple(genotype[config.edges:])

    assert check_strictly_increasing(stds), "model perturbation degree (standard deviation) should be monotonically increasing"
    criterion = nn.CrossEntropyLoss()

    train_dataloader_iter = iter(train_dataloader)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # NOTE: # iterations of BN statistics re-tracking for search loader
    max_train_iters = math.ceil(max_train_img_size / train_dataloader.batch_size)
    # NOTE: # iterations of measure validation accuracy for all validation images
    max_test_iters  = math.ceil(max_val_img_size / val_dataloader.batch_size)

    if max_train_iters > 0:
        # NOTE: [from SPOS paper] "Before the inference of an architecture, the statistics of all the Batch Normalization (BN) [9] operations are recalculated on a random subset of training data"
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean = torch.zeros_like(m.running_mean)
                m.running_var = torch.ones_like(m.running_var)

        model.train()

        for step in range(max_train_iters):
            batch = train_dataloader_iter.next()
            if len(batch) == 4:
                data, target, _, _ = batch
            elif len(batch) == 2:
                data, target = batch

            target = target.type(torch.LongTensor)
            data, target = data.to(device), target.to(device)
            output = model(data, normal_genotype, reduce_genotype)
            del data, target, output

    losses_per_stds = []

    model.eval()

    model_ = deepcopy(model)

    for std_idx, cur_std in enumerate(stds):
        val_dataloader_iter = iter(val_dataloader)
        
        # NOTE: adding gaussian noise parameterized by residual of std (\simga_t+1 - \sigma_t)in a cumulative way or direct adding way (\sigma_t)
        # NOTE: former bypasses deep copy of models each time, thus memory efficient.
        # NOTE: while former and latter could give different results, we take former for efficient memory usage.
        if std_idx == 0:
            std = cur_std # initial std
        else:
            std = cur_std - stds[std_idx - 1] # cumulate

        for name, param in model_.named_parameters():
            # NOTE: add gaussian noise for all parameters
            param.data.add_(torch.normal(0, std, size=param.size()).type(param.dtype).to(param.device))
        model_.eval()

        losses_over_batch = 0

        for step in range(max_test_iters):
            # NOTE: using validation dataset
            data, target = val_dataloader_iter.next()

            # NOTE: using search training dataset
            batchsize = data.shape[0]
            target = target.type(torch.LongTensor)
            data, target = data.to(device), target.to(device)

            logits = model_(data, normal_genotype, reduce_genotype)
            loss = criterion(logits, target)
            losses_over_batch += loss.item()

            del data, target, logits

        losses_mean = losses_over_batch / max_test_iters
        losses_per_stds.append(losses_mean)

    del model_

    # calculate wide & flat measure
    poor_minima = 0
    # NOTE: summation of gradients for loss. (regard non-perturbed loss value as 0)
    # TODO: add initial loss value (std:0, non-perturbed loss value)
    poor_minima += abs(losses_per_stds[0] / stds[0])
    for i in range(len(losses_per_stds) - 1):
        poor_minima += abs((losses_per_stds[i+1] - losses_per_stds[i]) / (stds[i+1] - stds[i]))
    
    wlm = 1 / (poor_minima + 1e-5) # wide & flat minima measure
    return wlm, losses_per_stds