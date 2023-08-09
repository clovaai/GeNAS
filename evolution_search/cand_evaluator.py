"""
Code Adapted from https://github.com/megvii-model/RLNAS/blob/main/darts_search_space/cifar10/rlnas/evolution_search/test_server.py
"""
import os
import torch
import time
from copy import deepcopy

from config import config
from super_model import Network
from operations import *
from metrics.tester_acc import get_cand_acc
from metrics.tester_wlm import get_cand_wlm

class CandEvaluator:
    def __init__(
        self,
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
    ):
        super().__init__()
        self.logging = logging
        self.final_model_path = final_model_path
        self.initial_model_path = initial_model_path
        self._recompile_net(
            self.final_model_path, self.initial_model_path
        )  # prepare initial model, final model

        self.metric = metric
        self.max_train_img_size = max_train_img_size
        self.max_val_img_size = max_val_img_size
        self.wlm_weight = wlm_weight
        self.acc_weight = acc_weight

        # data loader (required only when 'metric' == 'acc or 'wlm')
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # perturbations (required only when 'metric' contains 'wlm')
        self.perturb_stds = perturb_stds

        if (
            self.metric == "acc"
            or self.metric == "wlm"
            or self.metric == "acc+wlm"
            or self.metric == "angle+wlm"
            or self.metric == "angle+acc"
            or "angle+acc+sharp"
        ):
            assert self.train_loader
            assert self.valid_loader

        if (
            self.metric == "wlm"
            or self.metric == "acc+wlm"
            or self.metric == "angle+wlm"
            or self.metric == "angle+acc+sharp"
        ):
            assert self.perturb_stds

    def _recompile_net(self, final_model_path, initial_model_path):

        model = Network()
        initial_model = deepcopy(model)
        if torch.cuda.is_available():
            model = model.cuda()
            initial_model = initial_model.cuda()

        # model = nn.DataParallel(model)
        # initial_model = nn.DataParallel(initial_model)
        assert os.path.exists(final_model_path)
        self.logging.info("loading model {} ..........".format(final_model_path))
        checkpoint = torch.load(final_model_path, map_location="cpu")
        self.logging.info("loading states....")
        model.load_state_dict(checkpoint["state_dict"])
        self.logging.info(
            "loaded checkpoint {} epoch = {}".format(
                final_model_path, checkpoint["epoch"]
            )
        )

        assert os.path.exists(initial_model_path)
        self.logging.info(
            "loading initial model {} ..........".format(initial_model_path)
        )
        checkpoint = torch.load(initial_model_path, map_location="cpu")
        self.logging.info("loading states....")
        initial_model.load_state_dict(checkpoint["state_dict"])
        self.logging.info(
            "loaded checkpoint {} epoch = {}".format(
                initial_model_path, checkpoint["epoch"]
            )
        )

        self.model = model
        self.initial_model = initial_model

    def eval(self, cand):
        cand = list(
            cand
        )  # tuple to list. e.g) [4, 4, 3, 0, 6, 4, 0, 4, 0, 0, 2, 0, 0, 1, 4, 1, 7, 3, 0, 4, 6, 0, 0, 3, 0, 6, 0, 0] -> operation idx per (14 edges for normal cell + 14 edges for reduction cell)
        self.logging.info("cand={}".format(cand))
        res = self._test_candidate(cand)
        return res

    def _test_candidate(self, cand):
        res = dict()
        try:
            t0 = time.time()
            if self.metric == "angle":
                arch_score = self.get_angle(self.initial_model, self.model, cand)
            elif self.metric == "acc":
                arch_score = get_cand_acc(
                    self.model,
                    cand,
                    self.train_loader,
                    self.valid_loader,
                    self.max_train_img_size,
                    self.max_val_img_size,
                )[
                    0
                ]  # top-1 accuracy
            elif self.metric == "wlm":
                arch_score, _ = get_cand_wlm(
                    self.model,
                    cand,
                    self.train_loader,
                    self.valid_loader,
                    self.perturb_stds,
                    self.max_train_img_size,
                    self.max_val_img_size,
                )
            elif self.metric == "acc+wlm":
                gamma = self.wlm_weight
                beta = 1 / self.perturb_stds[0]
                # NOTE: get_cand_acc -> get_cand_wlm order matters.
                acc = get_cand_acc(
                    self.model, cand, self.train_loader, self.valid_loader, 15000, 25000
                )[
                    0
                ]  # top-1 accuracy
                wlm, _ = get_cand_wlm(
                    deepcopy(self.model),
                    cand,
                    self.train_loader,
                    self.valid_loader,
                    self.perturb_stds,
                    self.max_train_img_size,
                    self.max_val_img_size,
                )
                combined = acc + gamma * beta * wlm
                print(
                    "cal acc({acc}, {acc_percent}%) + {wlm_constant} * wlm({wlm}, {wlm_percent}%): {combined}".format(
                        acc=acc,
                        acc_percent=acc / combined * 100,
                        wlm_constant=gamma * beta,
                        wlm=gamma * beta * wlm,
                        wlm_percent=gamma * beta * wlm / combined * 100,
                        combined=combined,
                    )
                )
                arch_score = combined
            elif self.metric == "angle+wlm":
                gamma = self.wlm_weight
                beta = 1 / self.perturb_stds[0]
                # NOTE: get_cand_wlm -> get_angle order does not matter.
                wlm, _ = get_cand_wlm(
                    self.model,
                    cand,
                    self.train_loader,
                    self.valid_loader,
                    self.perturb_stds,
                    self.max_train_img_size,
                    self.max_val_img_size,
                )
                angle = self.get_angle(
                    self.initial_model, self.model, cand, normalize=True
                )  # 0~1 normalized Angle
                combined = angle + gamma * beta * wlm
                print(
                    "cal angle({angle}, {angle_percent}%) + {wlm_constant} * wlm({wlm}, {wlm_percent}%): {combined}".format(
                        angle=angle,
                        angle_percent=angle / combined * 100,
                        wlm_constant=gamma * beta,
                        wlm=gamma * beta * wlm,
                        wlm_percent=gamma * beta * wlm / combined * 100,
                        combined=combined,
                    )
                )
                arch_score = combined
            elif self.metric == "angle+acc":
                gamma = self.acc_weight
                # NOTE: get_cand_acc -> get_angle order does not matter.
                acc = get_cand_acc(
                    self.model,
                    cand,
                    self.train_loader,
                    self.valid_loader,
                    self.max_train_img_size,
                    self.max_val_img_size,
                )[
                    0
                ]  # top-1 accuracy
                angle = self.get_angle(
                    self.initial_model, self.model, cand, normalize=True
                )  # 0~1 normalized Angle
                combined = angle + gamma * acc
                print(
                    "cal angle({angle}, {angle_percent}%) + {acc_constant} * acc({acc}, {acc_percent}%): {combined}".format(
                        angle=angle,
                        angle_percent=angle / combined * 100,
                        acc_constant=gamma,
                        acc=gamma * acc,
                        acc_percent=gamma * acc / combined * 100,
                        combined=combined,
                    )
                )
                arch_score = combined

            self.logging.info("cand={}, {}={}".format(cand, self.metric, arch_score))
            self.logging.info("time: {}s".format(time.time() - t0))
            res = {"status": "success", self.metric: arch_score}
            return res
        except:
            import traceback

            traceback.print_exc()
            res["status"] = "failure"
            return res

    def get_arch_vector(self, model, normal_cand, reduction_cand):
        cand = []
        for layer in range(config.layers):
            if layer in [config.layers // 3, 2 * config.layers // 3]:
                cand.append(deepcopy(reduction_cand))
            else:
                cand.append(deepcopy(normal_cand))

        arch_vector, extra_params = [], []
        # Collect extra parameters
        stem = torch.cat(
            [
                model.stem[0].weight.data.reshape(-1),
                model.stem[1].weight.data.reshape(-1),
                model.stem[1].bias.data.reshape(-1),
            ]
        )
        extra_params += [stem]

        for i in range(len(model.cells)):
            # Collect extra parameters
            if isinstance(model.cells[i].preprocess0, FactorizedReduce):
                s0 = torch.cat(
                    [
                        model.cells[i].preprocess0.conv_1.weight.data.reshape(-1),
                        model.cells[i].preprocess0.conv_2.weight.data.reshape(-1),
                        model.cells[i].preprocess0.bn.weight.data.reshape(-1),
                        model.cells[i].preprocess0.bn.bias.data.reshape(-1),
                    ]
                )
            else:
                s0 = torch.cat(
                    [
                        model.cells[i].preprocess0.op[1].weight.data.reshape(-1),
                        model.cells[i].preprocess0.op[2].weight.data.reshape(-1),
                        model.cells[i].preprocess0.op[2].bias.data.reshape(-1),
                    ]
                )

            s1 = torch.cat(
                [
                    model.cells[i].preprocess1.op[1].weight.data.reshape(-1),
                    model.cells[i].preprocess1.op[2].weight.data.reshape(-1),
                    model.cells[i].preprocess1.op[2].bias.data.reshape(-1),
                ]
            )

            extra_params += [s0, s1]

            # Collect weight vecors of all paths
            param_list = []
            for path in config.paths:
                param_cell = []
                for index in range(1, len(path)):
                    j = path[index]
                    k = path[index - 1]
                    assert j >= 2
                    offset = 0
                    for tmp in range(2, j):
                        offset += tmp

                    if cand[i][k + offset] == config.NONE:  # None
                        param_cell = []
                        break

                    elif (
                        cand[i][k + offset] == config.MAX_POOLING_3x3
                        or cand[i][k + offset] == config.AVG_POOL_3x3
                    ):  # pooling
                        shape = (
                            model.cells[i]
                            ._ops[k + offset]
                            ._ops[4]
                            .op[1]
                            .weight.data.shape
                        )
                        shape = [shape[0], shape[2], shape[3]]
                        pooling_param = torch.ones(shape) * (1 / 9.0)
                        param_cell += [
                            torch.cat(
                                [
                                    pooling_param.reshape(-1).cuda(),
                                    model.cells[i]
                                    ._ops[k + offset]
                                    ._ops[cand[i][k + offset]][1]
                                    .weight.data.reshape(-1),
                                    model.cells[i]
                                    ._ops[k + offset]
                                    ._ops[cand[i][k + offset]][1]
                                    .bias.data.reshape(-1),
                                ]
                            )
                        ]
                    elif cand[i][k + offset] == config.SKIP_CONNECT:  # identity
                        # pass
                        if isinstance(
                            model.cells[i]._ops[k + offset]._ops[cand[i][k + offset]],
                            FactorizedReduce,
                        ):
                            param_cell += [
                                torch.cat(
                                    [
                                        model.cells[i]
                                        ._ops[k + offset]
                                        ._ops[cand[i][k + offset]]
                                        .conv_1.weight.data.reshape(-1),
                                        model.cells[i]
                                        ._ops[k + offset]
                                        ._ops[cand[i][k + offset]]
                                        .conv_2.weight.data.reshape(-1),
                                        model.cells[i]
                                        ._ops[k + offset]
                                        ._ops[cand[i][k + offset]]
                                        .bn.weight.data.reshape(-1),
                                        model.cells[i]
                                        ._ops[k + offset]
                                        ._ops[cand[i][k + offset]]
                                        .bn.bias.data.reshape(-1),
                                    ]
                                )
                            ]
                        elif isinstance(
                            model.cells[i]._ops[k + offset]._ops[cand[i][k + offset]],
                            Identity,
                        ):
                            shape = (
                                model.cells[i]
                                ._ops[k + offset]
                                ._ops[4]
                                .op[6]
                                .weight.data.shape
                            )
                            identity_param = torch.eye(shape[0], shape[0])
                            param_cell += [identity_param.reshape(-1).cuda()]
                        else:
                            raise Exception("Invalid operators !")

                    elif (
                        cand[i][k + offset] == config.SEP_CONV_3x3
                        or cand[i][k + offset] == config.SEP_CONV_5x5
                    ):  # sep conv
                        conv1 = torch.reshape(
                            model.cells[i]
                            ._ops[k + offset]
                            ._ops[cand[i][k + offset]]
                            .op[1]
                            .weight.data,
                            (-1,),
                        )
                        conv2 = torch.reshape(
                            model.cells[i]
                            ._ops[k + offset]
                            ._ops[cand[i][k + offset]]
                            .op[2]
                            .weight.data,
                            (-1,),
                        )
                        conv3 = torch.reshape(
                            model.cells[i]
                            ._ops[k + offset]
                            ._ops[cand[i][k + offset]]
                            .op[5]
                            .weight.data,
                            (-1,),
                        )
                        conv4 = torch.reshape(
                            model.cells[i]
                            ._ops[k + offset]
                            ._ops[cand[i][k + offset]]
                            .op[6]
                            .weight.data,
                            (-1,),
                        )
                        conv_cat = torch.cat([conv1, conv2, conv3, conv4])

                        bn1_weight = torch.reshape(
                            model.cells[i]
                            ._ops[k + offset]
                            ._ops[cand[i][k + offset]]
                            .op[3]
                            .weight.data,
                            (-1,),
                        )
                        bn1_bias = torch.reshape(
                            model.cells[i]
                            ._ops[k + offset]
                            ._ops[cand[i][k + offset]]
                            .op[3]
                            .bias.data,
                            (-1,),
                        )

                        bn2_weight = torch.reshape(
                            model.cells[i]
                            ._ops[k + offset]
                            ._ops[cand[i][k + offset]]
                            .op[7]
                            .weight.data,
                            (-1,),
                        )
                        bn2_bias = torch.reshape(
                            model.cells[i]
                            ._ops[k + offset]
                            ._ops[cand[i][k + offset]]
                            .op[7]
                            .bias.data,
                            (-1,),
                        )

                        bn_cat = torch.cat([bn1_weight, bn1_bias, bn2_weight, bn2_bias])

                        param_cell += [conv_cat]
                        param_cell += [bn_cat]

                    elif (
                        cand[i][k + offset] == config.DIL_CONV_3x3
                        or cand[i][k + offset] == config.DIL_CONV_5x5
                    ):
                        conv1 = torch.reshape(
                            model.cells[i]
                            ._ops[k + offset]
                            ._ops[cand[i][k + offset]]
                            .op[1]
                            .weight.data,
                            (-1,),
                        )
                        conv2 = torch.reshape(
                            model.cells[i]
                            ._ops[k + offset]
                            ._ops[cand[i][k + offset]]
                            .op[2]
                            .weight.data,
                            (-1,),
                        )
                        conv_cat = torch.cat([conv1, conv2])

                        bn1_weight = torch.reshape(
                            model.cells[i]
                            ._ops[k + offset]
                            ._ops[cand[i][k + offset]]
                            .op[3]
                            .weight.data,
                            (-1,),
                        )
                        bn1_bias = torch.reshape(
                            model.cells[i]
                            ._ops[k + offset]
                            ._ops[cand[i][k + offset]]
                            .op[3]
                            .weight.data,
                            (-1,),
                        )

                        bn_cat = torch.cat([bn1_weight, bn1_bias])

                        param_cell += [conv_cat]
                        param_cell += [bn_cat]
                    else:
                        raise Exception("Invalid operators !")

                # Get weight vector of a path
                if len(param_cell) != 0:
                    param_list.append(torch.cat(param_cell))

            # Get weight vector of a cell
            if len(param_list) != 0:
                arch_vector.append(torch.cat(param_list, dim=0))

        # Collect extra parameters
        extra_params.append(torch.reshape(model.classifier.weight.data, (-1,)))
        arch_vector += extra_params

        # Get weight vector of the whole model
        if len(arch_vector) != 0:
            arch_vector = torch.cat(arch_vector, dim=0)
        return arch_vector

    def get_angle(self, initial_model, model, cand, normalize=False):
        cosine = torch.nn.CosineSimilarity(dim=0).cuda()
        normal_cell = cand[: config.edges]
        redcution_cell = cand[config.edges :]
        vec1 = self.get_arch_vector(initial_model, normal_cell, redcution_cell)
        vec2 = self.get_arch_vector(model, normal_cell, redcution_cell)
        if normalize:
            return (
                torch.acos(cosine(vec1, vec2)).cpu().item()
                / torch.acos(torch.Tensor([-1])).cpu().item()
            )
        else:
            return torch.acos(cosine(vec1, vec2)).cpu().item()
