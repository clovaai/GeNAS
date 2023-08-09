"""
GeNAS
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""
import torch
import math
from config import config
assert torch.cuda.is_available()

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def no_grad_wrapper(func):
    def new_func(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return new_func

@no_grad_wrapper
def get_cand_acc(model, genotype, train_dataloader, val_dataloader, max_train_img_size, max_val_img_size=25000):
    '''
        genotype: normal (14 edges) + reduce cell (14 edges) with operation indices. e.g. 6, 6, 3, 0, 4, 7, 0, 0, 1, 0, 6, 0, 6, 0, 1, 4, 3, 7, 0, 7, 0, 0, 4, 0, 0, 0, 3, 2]
        train_dataloader: half (25K) of original training set (50K).
        val_dataloader: another half (25K) of original training set (50K).
    '''
    # separate genotype
    normal_genotype = tuple(genotype[:config.edges])
    reduce_genotype = tuple(genotype[config.edges:])

    train_dataloader_iter = iter(train_dataloader)
    val_dataloader_iter = iter(val_dataloader)

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

    top1 = 0
    top5 = 0
    total = 0

    print('starting test....')
    model.eval()

    for step in range(max_test_iters):
        data, target = val_dataloader_iter.next()
        batchsize = data.shape[0]
        target = target.type(torch.LongTensor)
        data, target = data.to(device), target.to(device)
        logits = model(data, normal_genotype, reduce_genotype)
        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        top1 += prec1.item() * batchsize
        top5 += prec5.item() * batchsize
        total += batchsize

        del data, target, logits, prec1, prec5

    top1, top5 = top1 / total, top5 / total
    top1, top5 = top1 / 100, top5 / 100
    return top1, top5

def main():
    pass