'''
Code adapted from https://github.com/megvii-model/RLNAS/blob/main/darts_search_space/cifar10/rlnas/train_supernet/train.py
'''
import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from super_model import Network
from copy import deepcopy
from config import config
from datasets import get_datasets, get_nas_search_loaders

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='models', help='experiment name')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--random_label', type=int, choices=[0, 1], default=1, help='Whether use random label for dataset or not. (default: True)')
parser.add_argument('--split_data', type=int, choices=[0, 1], default=1, help='Whether use split data for training & validation. (default: True)')
args = parser.parse_args()

args.split_data = bool(args.split_data)

utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 100

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  seed = args.seed
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  # NOTE: layers: number of cells in network
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  if args.split_data: # NOTE: split train data in half to be new train, val set. new train is used for supernet training, new val set is used for evaluation
    train_data, valid_data, xshape, class_num = get_datasets('cifar100', args.data, -1, args.seed, random_label=bool(args.random_label))
    train_queue, _, _, valid_queue = get_nas_search_loaders(train_data, valid_data, 'cifar100',
                                                            'datasets/configs/', \
                                                            (args.batch_size, args.batch_size), 4, use_valid_no_shuffle=True)
  else:
    assert ValueError("only --split_data 1 is supported")                                                    

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  operations = []
  for _ in range(config.edges):
      operations.append(list(range(config.op_num)))
  print('operations={}'.format(operations))

  utils.save_checkpoint({'epoch': -1,
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict()}, args.save)

  for epoch in range(args.epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    # training
    seed, train_acc, train_obj = train(train_queue, model, criterion, optimizer, operations, seed, epoch)
    logging.info('train_acc %f', train_acc)

    # validation
    valid_acc, valid_obj = infer(valid_queue, model, criterion, seed, operations)
    logging.info('valid_acc %f', valid_acc)

    if (epoch+1)%5 == 0:
      utils.save_checkpoint({'epoch':epoch,
                             'state_dict':model.state_dict(),
                             'optimizer':optimizer.state_dict()}, args.save)

def get_random_cand(seed, operations):
  # Uniform Sampling
  rng = []
  for op in operations:
    np.random.seed(seed)
    k = np.random.randint(len(op))
    select_op = op[k]
    rng.append(select_op)
    seed += 1

  return rng, seed

def train(train_queue, model, criterion, optimizer, operations, seed, epoch):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  model.train()

  for step, batch in enumerate(train_queue):
    if len(batch) == 4:
      input, target, _, _ = batch
    elif len(batch) == 2:
      input, target = batch
    n = input.size(0)

    input = input.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True)

    optimizer.zero_grad()

    # NOTE: per training iteration, operation per edge is randomly sampled.. (as in SPOS!. no architecture parameters.)
    normal_rng, seed = get_random_cand(seed, operations)
    reduction_rng, seed = get_random_cand(seed, operations)

    normal_rng = utils.check_cand(normal_rng, operations)
    reduction_rng = utils.check_cand(reduction_rng, operations)

    logits = model(input, normal_rng, reduction_rng)
    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return seed, top1.avg, objs.avg


def infer(valid_queue, model, criterion, seed, operations):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  normal_rng, seed = get_random_cand(seed, operations)
  reduction_rng, seed = get_random_cand(seed, operations)

  normal_rng = utils.check_cand(normal_rng, operations)
  reduction_rng = utils.check_cand(reduction_rng, operations)

  # NOTE: no optimize for architecture parameters (abscence of architecture parameters)
  # NOTE: instead, randomly select operation for each edge and evaluate.
  for step, (input, target) in enumerate(valid_queue):
    input = input.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True)

    with torch.no_grad():
      logits = model(input, normal_rng, reduction_rng)
      loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main()
