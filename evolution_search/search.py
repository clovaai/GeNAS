"""
Code Adapted from https://github.com/megvii-model/RLNAS/blob/main/darts_search_space/cifar10/rlnas/evolution_search/search.py
"""

import os
import sys
import time
import glob
import random
import numpy as np
import pickle
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from cand_evaluator import CandEvaluator
from genotypes import parse_searched_cell
from datasets import get_datasets, get_nas_search_loaders
from config import config
import collections
import sys

sys.setrecursionlimit(10000)
import argparse
import utils
import functools

print = functools.partial(print, flush=True)

choice = (
    lambda x: x[np.random.randint(len(x))] if isinstance(x, tuple) else choice(tuple(x))
)


class EvolutionTrainer(object):
    def __init__(
        self,
        log_dir,
        final_model_path,
        initial_model_path,
        metric="angle",
        train_loader=None,
        valid_loader=None,
        perturb_stds=None,
        max_train_img_size=5000,
        max_val_img_size=10000,
        wlm_weight=0,
        acc_weight=0,
        refresh=False,
    ):
        self.log_dir = log_dir
        self.checkpoint_name = os.path.join(self.log_dir, "checkpoint.brainpkl")
        self.refresh = refresh
        self.cand_evaluator = CandEvaluator(
            logging,
            final_model_path,
            initial_model_path,
            metric,
            train_loader,
            valid_loader,
            perturb_stds,
            max_train_img_size,
            max_val_img_size,
            wlm_weight,
            acc_weight,
        )

        self.memory = []
        self.candidates = []
        self.vis_dict = {}
        self.keep_top_k = {config.select_num: [], 50: []}
        self.epoch = 0
        self.cand_idx = 0  # for generating candidate idx
        self.operations = [list(range(config.op_num)) for _ in range(config.edges)]

        self.metric = metric

    def save_checkpoint(self):
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        info = {}
        info["memory"] = self.memory
        info["candidates"] = self.candidates
        info["vis_dict"] = self.vis_dict
        info["keep_top_k"] = self.keep_top_k
        info["epoch"] = self.epoch
        info["cand_idx"] = self.cand_idx
        torch.save(info, self.checkpoint_name)
        logging.info("save checkpoint to {}".format(self.checkpoint_name))

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_name):
            return False
        info = torch.load(self.checkpoint_name)
        self.memory = info["memory"]
        self.candidates = info["candidates"]
        self.vis_dict = info["vis_dict"]
        self.keep_top_k = info["keep_top_k"]
        self.epoch = info["epoch"]
        self.cand_idx = info["cand_idx"]

        if self.refresh:
            for i, j in self.vis_dict.items():
                for k in ["test_key"]:
                    if k in j:
                        j.pop(k)
            self.refresh = False

        logging.info("load checkpoint from {}".format(self.checkpoint_name))
        return True

    def legal(self, cand):
        assert isinstance(cand, tuple) and len(cand) == (2 * config.edges)
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if "visited" in info:
            return False

        if config.flops_limit is not None:
            pass

        self.vis_dict[cand] = info
        info["visited"] = True

        return True

    def update_top_k(self, candidates, *, k, key, reverse=False):
        assert k in self.keep_top_k
        logging.info("select ......")
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]

    def gen_key(self, cand):
        # NOTE: generate unique id for candidate
        self.cand_idx += 1
        key = "{}-{}".format(self.cand_idx, time.time())
        return key

    def eval_cand(self, cand, cand_key):
        # NOTE: evaluate candidate
        try:
            result = self.cand_evaluator.eval(cand)
            return result
        except:
            import traceback

            traceback.print_exc()
            return {"status": "uncatched error"}

    def sync_candidates(self):
        while True:
            ok = True
            for cand in self.candidates:
                info = self.vis_dict[cand]
                if self.metric in info:
                    continue
                ok = False
                if "test_key" not in info:
                    info["test_key"] = self.gen_key(cand)

            self.save_checkpoint()

            for cand in self.candidates:
                info = self.vis_dict[cand]
                if self.metric in info:
                    continue
                key = info.pop("test_key")

                try:
                    logging.info("try to get {}".format(key))
                    res = self.eval_cand(
                        cand, key
                    )  # NOTE: currently, key and cand has implicit connection
                    logging.info(res)
                    info[self.metric] = res[self.metric]
                    self.save_checkpoint()
                except:
                    import traceback

                    traceback.print_exc()
                    time.sleep(1)

            time.sleep(5)
            if ok:
                break

    def stack_random_cand(self, random_func, *, batchsize=10):
        while True:
            cands = [random_func() for _ in range(batchsize)]
            for cand in cands:
                if cand not in self.vis_dict:
                    self.vis_dict[cand] = {}
                else:
                    continue
                info = self.vis_dict[cand]
                # for cand in cands:
                yield cand

    def stack_random_cand_crossover(self, random_func, max_iters, *, batchsize=10):
        cand_count = 0
        while True:
            if cand_count > max_iters:
                break
            cands = [random_func() for _ in range(batchsize)]
            cand_count += 1
            for cand in cands:
                if cand not in self.vis_dict:
                    self.vis_dict[cand] = {}
                else:
                    continue
                info = self.vis_dict[cand]
                # for cand in cands:
                yield cand

    def random_can(self, num):
        logging.info("random select ........")
        candidates = []
        cand_iter = self.stack_random_cand(
            lambda: tuple(
                np.random.randint(config.op_num) for _ in range(2 * config.edges)
            )
        )
        while len(candidates) < num:
            cand = next(cand_iter)
            normal_cand = cand[: config.edges]
            reduction_cand = cand[config.edges :]
            normal_cand = utils.check_cand(normal_cand, self.operations)
            reduction_cand = utils.check_cand(reduction_cand, self.operations)
            cand = normal_cand + reduction_cand
            cand = tuple(cand)
            if not self.legal(cand):
                continue
            candidates.append(cand)
            logging.info("random {}/{}".format(len(candidates), num))
        logging.info("random_num = {}".format(len(candidates)))
        return candidates

    def get_mutation(self, k, mutation_num, m_prob):
        assert k in self.keep_top_k
        logging.info("mutation ......")
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = list(choice(self.keep_top_k[k]))
            for i in range(config.edges):
                if np.random.random_sample() < m_prob:
                    cand[i] = np.random.randint(0, config.op_num)
            return tuple(cand)

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < mutation_num and max_iters > 0:
            cand = next(cand_iter)
            normal_cand = cand[: config.edges]
            reduction_cand = cand[config.edges :]
            normal_cand = utils.check_cand(normal_cand, self.operations)
            reduction_cand = utils.check_cand(reduction_cand, self.operations)
            cand = normal_cand + reduction_cand
            cand = tuple(cand)
            if not self.legal(cand):
                continue
            res.append(cand)
            logging.info("mutation {}/{}".format(len(res), mutation_num))
            max_iters -= 1

        logging.info("mutation_num = {}".format(len(res)))
        return res

    def get_crossover(self, k, crossover_num):
        assert k in self.keep_top_k
        logging.info("crossover ......")
        res = []
        iter = 0
        max_iters = 10 * crossover_num

        def random_func():
            p1 = choice(self.keep_top_k[k])
            p2 = choice(self.keep_top_k[k])
            return tuple(choice([i, j]) for i, j in zip(p1, p2))

        cand_iter = self.stack_random_cand_crossover(random_func, crossover_num)
        while len(res) < crossover_num:
            try:
                cand = next(cand_iter)
                normal_cand = cand[: config.edges]
                reduction_cand = cand[config.edges :]
                normal_cand = utils.check_cand(normal_cand, self.operations)
                reduction_cand = utils.check_cand(reduction_cand, self.operations)
                cand = normal_cand + reduction_cand
                cand = tuple(cand)
            except Exception as e:
                logging.info(e)
                break
            if not self.legal(cand):
                continue
            res.append(cand)
            logging.info("crossover {}/{}".format(len(res), crossover_num))

        logging.info("crossover_num = {}".format(len(res)))
        return res

    def train(self):
        logging.info(
            "population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}".format(
                config.population_num,
                config.select_num,
                config.mutation_num,
                config.crossover_num,
                config.population_num - config.mutation_num - config.crossover_num,
                config.max_epochs,
            )
        )

        if not self.load_checkpoint():
            self.candidates = self.random_can(config.population_num)
            self.save_checkpoint()

        while self.epoch < config.max_epochs:
            logging.info("epoch = {}".format(self.epoch))

            self.sync_candidates()  # NOTE: evaluate candidates

            logging.info("sync finish")

            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)
                self.vis_dict[cand]["visited"] = True

            self.update_top_k(
                self.candidates,
                k=config.select_num,
                key=lambda x: self.vis_dict[x][self.metric],
                reverse=True,
            )
            self.update_top_k(
                self.candidates,
                k=50,
                key=lambda x: self.vis_dict[x][self.metric],
                reverse=True,
            )

            logging.info(
                "epoch = {} : top {} result".format(
                    self.epoch, len(self.keep_top_k[50])
                )
            )
            for i, cand in enumerate(self.keep_top_k[50]):
                logging.info(
                    "No.{} {} {} = {}".format(
                        i + 1, cand, self.metric, self.vis_dict[cand][self.metric]
                    )
                )
                # ops = [config.blocks_keys[i] for i in cand]
                ops = [config.blocks_keys[i] for i in cand]
                logging.info(ops)

            mutation = self.get_mutation(
                config.select_num, config.mutation_num, config.m_prob
            )
            crossover = self.get_crossover(config.select_num, config.crossover_num)
            rand = self.random_can(
                config.population_num - len(mutation) - len(crossover)
            )
            self.candidates = mutation + crossover + rand

            self.epoch += 1
            self.save_checkpoint()

        logging.info(self.keep_top_k[config.select_num])
        logging.info("finish!")
        logging.info(
            "Top-1 Searched Cell Architecture : {}".format(
                parse_searched_cell(self.keep_top_k[config.select_num][0])
            )
        )


def prepare_seed(rand_seed):
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)


class SplitArgs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, [float(val) for val in values.split(",")])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--refresh", action="store_true")
    parser.add_argument("--save", type=str, default="log", help="experiment name")
    parser.add_argument("--seed", type=int, default=1, help="experiment name")
    parser.add_argument(
        "--init_model_path",
        type=str,
        default=config.initial_net_cache,
        help="initial model ckpt path",
    )
    parser.add_argument(
        "--model_path", type=str, default=config.net_cache, help="final model ckpt path"
    )
    parser.add_argument(
        "--metric", type=str, default="angle", help="metric to evaulate candidate with."
    )

    """ below are required if args.metric is not "angle" """
    parser.add_argument(
        "--data",
        type=str,
        default="",
        help='data root path. required if --metric is not "angle"',
    )
    parser.add_argument(
        "--split_data",
        type=int,
        choices=[0, 1],
        default=1,
        help="Whether use split data for training & validation. (default: True)",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="train batch_size")
    parser.add_argument(
        "--test_batch_size", type=int, default=512, help="test batch_size"
    )
    parser.add_argument(
        "--cutout", action="store_true", default=False, help="use cutout"
    )
    parser.add_argument("--cutout_length", type=int, default=16, help="cutout length")
    # GeNAS hyperparameters
    parser.add_argument(
        "--stds",
        default=None,
        action=SplitArgs,
        help="std values for weight perturbation",
    )
    parser.add_argument(
        "--max_train_img_size",
        type=int,
        default=5000,
        help="maximum number of training imgs for batch norm statistics recalculation.",
    )
    parser.add_argument(
        "--max_val_img_size",
        type=int,
        default=10000,
        help="maximum number of validation imgs for evaluating architecture candidates. (required only for wlm)",
    )
    # Combined metric
    parser.add_argument("--wlm_weight", type=float, default=0, help="wlm weight")
    parser.add_argument("--acc_weight", type=float, default=0, help="acc weight")

    args = parser.parse_args()

    args.split_data = bool(args.split_data)

    utils.create_exp_dir(args.save, scripts_to_save=glob.glob("*.py"))

    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format=log_format,
        datefmt="%m/%d %I:%M:%S %p",
    )
    fh = logging.FileHandler(os.path.join(args.save, "search_log.txt"))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    if (
        args.split_data
    ):  # NOTE: split train data in half to be new train, val set. new train is used for supernet training, new val set is used for evaluation
        train_data, valid_data, xshape, class_num = get_datasets(
            "cifar100", args.data, -1, args.seed, random_label=False
        )  # NOTE: using GT label
        train_queue, _, _, valid_queue = get_nas_search_loaders(
            train_data,
            valid_data,
            "cifar100",
            "datasets/configs/",
            (args.batch_size, args.batch_size),
            4,
            use_valid_no_shuffle=True,
        )
    else:
        assert ValueError("only --split_data 1 is supported")

    refresh = args.refresh
    # np.random.seed(args.seed)
    prepare_seed(args.seed)

    t = time.time()

    trainer = EvolutionTrainer(
        args.save,
        args.model_path,
        args.init_model_path,
        metric=args.metric,
        train_loader=train_queue,
        valid_loader=valid_queue,
        perturb_stds=args.stds,
        max_train_img_size=args.max_train_img_size,
        max_val_img_size=args.max_val_img_size,
        wlm_weight=args.wlm_weight,
        acc_weight=args.acc_weight,
        refresh=refresh,
    )

    trainer.train()
    logging.info("total searching time = {:.2f} hours".format((time.time() - t) / 3600))


if __name__ == "__main__":
    try:
        main()
        os._exit(0)
    except:
        import traceback

        traceback.print_exc()
        time.sleep(1)
        os._exit(1)
