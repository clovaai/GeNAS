'''
Copied from https://github.com/megvii-model/RLNAS/blob/main/darts_search_space/cifar10/rlnas/retrain_architecture/utils.py
'''

import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2
import random
import PIL
from PIL import Image
import math

class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].reshape(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)


class OpencvResize(object):

    def __init__(self, size=256):
        self.size = size

    def __call__(self, img):
        assert isinstance(img, PIL.Image.Image)
        img = np.asarray(img)  # (H,W,3) RGB
        img = img[:, :, ::-1]  # 2 BGR
        img = np.ascontiguousarray(img)
        H, W, _ = img.shape
        target_size = (int(self.size / H * W + 0.5), self.size) if H < W else (self.size, int(self.size / W * H + 0.5))
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        img = img[:, :, ::-1]  # 2 RGB
        img = np.ascontiguousarray(img)
        img = Image.fromarray(img)
        return img

class RandomResizedCrop(object):

    def __init__(self, scale=(0.08, 1.0), target_size: int = 224, max_attempts: int = 10):
        assert scale[0] <= scale[1]
        self.scale = scale
        assert target_size > 0
        self.target_size = target_size
        assert max_attempts > 0
        self.max_attempts = max_attempts

    def __call__(self, img):
        assert isinstance(img, PIL.Image.Image)
        img = np.asarray(img, dtype=np.uint8)
        H, W, C = img.shape

        well_cropped = False
        for _ in range(self.max_attempts):
            crop_area = (H * W) * random.uniform(self.scale[0], self.scale[1])
            crop_edge = round(math.sqrt(crop_area))
            dH = H - crop_edge
            dW = W - crop_edge
            crop_left = random.randint(min(dW, 0), max(dW, 0))
            crop_top = random.randint(min(dH, 0), max(dH, 0))
            if dH >= 0 and dW >= 0:
                well_cropped = True
                break

        crop_bottom = crop_top + crop_edge
        crop_right = crop_left + crop_edge
        if well_cropped:
            crop_image = img[crop_top:crop_bottom, :, :][:, crop_left:crop_right, :]

        else:
            roi_top = max(crop_top, 0)
            padding_top = roi_top - crop_top
            roi_bottom = min(crop_bottom, H)
            padding_bottom = crop_bottom - roi_bottom
            roi_left = max(crop_left, 0)
            padding_left = roi_left - crop_left
            roi_right = min(crop_right, W)
            padding_right = crop_right - roi_right

            roi_image = img[roi_top:roi_bottom, :, :][:, roi_left:roi_right, :]
            crop_image = cv2.copyMakeBorder(roi_image, padding_top, padding_bottom, padding_left, padding_right,
                                            borderType=cv2.BORDER_CONSTANT, value=0)

        random.choice([1])
        target_image = cv2.resize(crop_image, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
        target_image = PIL.Image.fromarray(target_image.astype('uint8'))
        return target_image


class LighteningJitter(object):

    def __init__(self, eigen_vecs, eigen_values, max_eigen_jitter=0.1):
        self.eigen_vecs = np.array(eigen_vecs, dtype=np.float32)
        self.eigen_values = np.array(eigen_values, dtype=np.float32)
        self.max_eigen_jitter = max_eigen_jitter

    def __call__(self, img):
        assert isinstance(img, PIL.Image.Image)
        img = np.asarray(img, dtype=np.float32)
        img = np.ascontiguousarray(img / 255)

        cur_eigen_jitter = np.random.normal(scale=self.max_eigen_jitter, size=self.eigen_values.shape)
        color_purb = (self.eigen_vecs @ (self.eigen_values * cur_eigen_jitter)).reshape([1, 1, -1])
        img += color_purb
        img = np.ascontiguousarray(img * 255)
        img.clip(0, 255, out=img)
        img = PIL.Image.fromarray(np.uint8(img))
        return img

def get_train_transform():
    eigvec = np.array([
        [-0.5836, -0.6948, 0.4203],
        [-0.5808, -0.0045, -0.8140],
        [-0.5675, 0.7192, 0.4009]
    ])

    eigval = np.array([0.2175, 0.0188, 0.0045])

    transform = transforms.Compose([
        RandomResizedCrop(target_size=224, scale=(0.08, 1.0)),
        LighteningJitter(eigen_vecs=eigvec[::-1, :], eigen_values=eigval,
                         max_eigen_jitter=0.1),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
    ])
    return transform

def get_eval_transform():
    transform = transforms.Compose([
                    OpencvResize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
    ])

    return transform