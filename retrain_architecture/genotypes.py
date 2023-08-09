'''
Code adapted from https://github.com/megvii-model/RLNAS/blob/main/darts_search_space/cifar10/rlnas/retrain_architecture/genotypes.py
'''

from ast import parse
from collections import namedtuple

Genotype = namedtuple("Genotype", "normal normal_concat reduce reduce_concat")

PRIMITIVES = [
    "none",
    "max_pool_3x3",
    "avg_pool_3x3",
    "skip_connect",
    "sep_conv_3x3",
    "sep_conv_5x5",
    "dil_conv_3x3",
    "dil_conv_5x5",
]

NASNet = Genotype(
    normal=[
        ("sep_conv_5x5", 1),
        ("sep_conv_3x3", 0),
        ("sep_conv_5x5", 0),
        ("sep_conv_3x3", 0),
        ("avg_pool_3x3", 1),
        ("skip_connect", 0),
        ("avg_pool_3x3", 0),
        ("avg_pool_3x3", 0),
        ("sep_conv_3x3", 1),
        ("skip_connect", 1),
    ],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[
        ("sep_conv_5x5", 1),
        ("sep_conv_7x7", 0),
        ("max_pool_3x3", 1),
        ("sep_conv_7x7", 0),
        ("avg_pool_3x3", 1),
        ("sep_conv_5x5", 0),
        ("skip_connect", 3),
        ("avg_pool_3x3", 2),
        ("sep_conv_3x3", 2),
        ("max_pool_3x3", 1),
    ],
    reduce_concat=[4, 5, 6],
)

AmoebaNet = Genotype(
    normal=[
        ("avg_pool_3x3", 0),
        ("max_pool_3x3", 1),
        ("sep_conv_3x3", 0),
        ("sep_conv_5x5", 2),
        ("sep_conv_3x3", 0),
        ("avg_pool_3x3", 3),
        ("sep_conv_3x3", 1),
        ("skip_connect", 1),
        ("skip_connect", 0),
        ("avg_pool_3x3", 1),
    ],
    normal_concat=[4, 5, 6],
    reduce=[
        ("avg_pool_3x3", 0),
        ("sep_conv_3x3", 1),
        ("max_pool_3x3", 0),
        ("sep_conv_7x7", 2),
        ("sep_conv_7x7", 0),
        ("avg_pool_3x3", 1),
        ("max_pool_3x3", 0),
        ("max_pool_3x3", 1),
        ("conv_7x1_1x7", 0),
        ("sep_conv_3x3", 5),
    ],
    reduce_concat=[3, 4, 6],
)

DARTS_V1_CIFAR10 = Genotype(
    normal=[
        ("sep_conv_3x3", 1),
        ("sep_conv_3x3", 0),
        ("skip_connect", 0),
        ("sep_conv_3x3", 1),
        ("skip_connect", 0),
        ("sep_conv_3x3", 1),
        ("sep_conv_3x3", 0),
        ("skip_connect", 2),
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ("max_pool_3x3", 0),
        ("max_pool_3x3", 1),
        ("skip_connect", 2),
        ("max_pool_3x3", 0),
        ("max_pool_3x3", 0),
        ("skip_connect", 2),
        ("skip_connect", 2),
        ("avg_pool_3x3", 0),
    ],
    reduce_concat=[2, 3, 4, 5],
)

DARTS_V2_CIFAR10 = Genotype(
    normal=[
        ("sep_conv_3x3", 0),
        ("sep_conv_3x3", 1),
        ("sep_conv_3x3", 0),
        ("sep_conv_3x3", 1),
        ("sep_conv_3x3", 1),
        ("skip_connect", 0),
        ("skip_connect", 0),
        ("dil_conv_3x3", 2),
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ("max_pool_3x3", 0),
        ("max_pool_3x3", 1),
        ("skip_connect", 2),
        ("max_pool_3x3", 1),
        ("max_pool_3x3", 0),
        ("skip_connect", 2),
        ("skip_connect", 2),
        ("max_pool_3x3", 1),
    ],
    reduce_concat=[2, 3, 4, 5],
)


def parse_searched_cell(normal_reduce_cell):
    """
    normal_reduce_cell: list of normal + reduce cell.
    e.g) [
        14 elements for normal cell edges where each element denote operation for each edge +
        14 elements for reduce cell edges where each element denote operation for each edge
    ]
    """
    assert (
        len(normal_reduce_cell) == 28
    ), "cell should contain normal + reduce edges (14 + 14 = 28)"
    normal_cell = normal_reduce_cell[:14]
    reduce_cell = normal_reduce_cell[14:]

    normal_cell_decoded = []
    reduce_cell_decoded = []
    # normal cell decode
    for i in range(len(normal_cell)):
        # NOTE: for generating intermediate node 0
        if i in [0, 1]:
            normal_cell_decoded.append((PRIMITIVES[normal_cell[i]], i))
        # NOTE: for generating intermediate node 1
        elif i in [2, 3, 4]:
            if normal_cell[i] != 0:
                normal_cell_decoded.append((PRIMITIVES[normal_cell[i]], i - 2))
        # NOTE: for generating intermediate node 2
        elif i in [5, 6, 7, 8]:
            if normal_cell[i] != 0:
                normal_cell_decoded.append((PRIMITIVES[normal_cell[i]], i - 5))
            # NOTE: for generating intermediate node 3
        elif i in [9, 10, 11, 12, 13]:
            if normal_cell[i] != 0:
                normal_cell_decoded.append((PRIMITIVES[normal_cell[i]], i - 9))

    # reduce cell decode
    for i in range(len(reduce_cell)):
        # NOTE: for generating intermediate node 0
        if i in [0, 1]:
            reduce_cell_decoded.append((PRIMITIVES[reduce_cell[i]], i))
        # NOTE: for generating intermediate node 1
        elif i in [2, 3, 4]:
            if reduce_cell[i] != 0:
                reduce_cell_decoded.append((PRIMITIVES[reduce_cell[i]], i - 2))
        # NOTE: for generating intermediate node 2
        elif i in [5, 6, 7, 8]:
            if reduce_cell[i] != 0:
                reduce_cell_decoded.append((PRIMITIVES[reduce_cell[i]], i - 5))
            # NOTE: for generating intermediate node 3
        elif i in [9, 10, 11, 12, 13]:
            if reduce_cell[i] != 0:
                reduce_cell_decoded.append((PRIMITIVES[reduce_cell[i]], i - 9))

    return Genotype(
        normal=normal_cell_decoded,
        normal_concat=[2, 3, 4, 5],
        reduce=reduce_cell_decoded,
        reduce_concat=[2, 3, 4, 5],
    )


# print(parse_searched_cell((5, 4, 6, 0, 5, 4, 0, 0, 7, 0, 7, 6, 0, 0, 5, 6, 4, 0, 5, 0, 6, 0, 4, 0, 1, 5, 0, 0)))
# cand=[[5, 4, 6, 0, 5, 4, 0, 0, 7, 0, 7, 6, 0, 0], [5, 6, 4, 0, 5, 0, 6, 0, 4, 0, 1, 5, 0, 0]]
# NOTE: [5, 4, 6, 0, 5, 4, 0, 0, 7, 0, 7, 6, 0, 0] means
# NOTE: [5, 4]: for generating intermediate node 0, operation 5:sep_conv_5x5(k-2 node(prev prev cell output)) + operation 4: sep_conv_3x3(k-1 node (prev cell output))
# NOTE: [6, 0, 5]: for generating intermediate node 1, operation 6:dil_conv_3x3(itm node 0) + operation 5:sep_conv_5x5(itm node 2)
# NOTE: [4, 0, 0, 7]: for generating intermediate node 2, operation 4:sep_conv_3x3(itm node 0) + operation 7:dil_conv_5x5(itm node 3)
# NOTE: [0, 7, 6, 0, 0]: for generating intermediate node 3, operation 7:dil_conv_5x5(itm node 1) + operation 6:dil_conv_3x3(itm node 2)
# NOTE: all intermediate node outputs (0, 1, 2, 3) are concatenated to be the output of current cell.
# RLDARTS = Genotype(
#     normal=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0),
#             ('dil_conv_5x5', 3), ('dil_conv_5x5', 1), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5],
#     reduce=[('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('dil_conv_3x3', 1),
#             ('sep_conv_3x3', 3), ('max_pool_3x3', 1), ('sep_conv_5x5', 2)], reduce_concat=[2, 3, 4, 5])
RLDARTS_OURS_GT = parse_searched_cell(
    (5, 5, 2, 5, 0, 4, 4, 0, 0, 0, 4, 0, 0, 4, 3, 3, 0, 3, 2, 3, 7, 0, 0, 3, 0, 0, 0, 4)
)
PCDARTS_OURS_SEARCHEPOCH40 = Genotype(
    normal=[
        ("sep_conv_3x3", 0),
        ("sep_conv_3x3", 1),
        ("sep_conv_3x3", 1),
        ("sep_conv_5x5", 0),
        ("sep_conv_5x5", 1),
        ("sep_conv_3x3", 3),
        ("sep_conv_5x5", 4),
        ("sep_conv_3x3", 0),
    ],
    normal_concat=range(2, 6),
    reduce=[
        ("max_pool_3x3", 0),
        ("sep_conv_3x3", 1),
        ("max_pool_3x3", 0),
        ("skip_connect", 1),
        ("sep_conv_5x5", 2),
        ("sep_conv_3x3", 0),
        ("skip_connect", 0),
        ("dil_conv_3x3", 4),
    ],
    reduce_concat=range(2, 6),
)

PCDARTS_OURS = Genotype(
    normal=[
        ("dil_conv_5x5", 1),
        ("dil_conv_3x3", 0),
        ("dil_conv_5x5", 1),
        ("max_pool_3x3", 0),
        ("dil_conv_3x3", 0),
        ("sep_conv_3x3", 3),
        ("sep_conv_3x3", 0),
        ("sep_conv_3x3", 2),
    ],
    normal_concat=range(2, 6),
    reduce=[
        ("max_pool_3x3", 0),
        ("max_pool_3x3", 1),
        ("max_pool_3x3", 0),
        ("skip_connect", 2),
        ("sep_conv_5x5", 2),
        ("skip_connect", 1),
        ("sep_conv_3x3", 0),
        ("sep_conv_3x3", 2),
    ],
    reduce_concat=range(2, 6),
)

# PDARTS searched on CIFAR-10
PDARTS_CIFAR10 = Genotype(
    normal=[
        ("skip_connect", 0),
        ("dil_conv_3x3", 1),
        ("skip_connect", 0),
        ("sep_conv_3x3", 1),
        ("sep_conv_3x3", 1),
        ("sep_conv_3x3", 3),
        ("sep_conv_3x3", 0),
        ("dil_conv_5x5", 4),
    ],
    normal_concat=range(2, 6),
    reduce=[
        ("avg_pool_3x3", 0),
        ("sep_conv_5x5", 1),
        ("sep_conv_3x3", 0),
        ("dil_conv_5x5", 2),
        ("max_pool_3x3", 0),
        ("dil_conv_3x3", 1),
        ("dil_conv_3x3", 1),
        ("dil_conv_5x5", 3),
    ],
    reduce_concat=range(2, 6),
)


# DARTS-v1 searched on CIFAR-100
DARTS_V1_CIFAR100 = Genotype(
    normal=[
        ("skip_connect", 0),
        ("sep_conv_3x3", 1),
        ("skip_connect", 0),
        ("sep_conv_3x3", 1),
        ("skip_connect", 0),
        ("skip_connect", 1),
        ("skip_connect", 0),
        ("skip_connect", 1),
    ],
    normal_concat=range(2, 6),
    reduce=[
        ("avg_pool_3x3", 0),
        ("avg_pool_3x3", 1),
        ("avg_pool_3x3", 0),
        ("skip_connect", 2),
        ("skip_connect", 2),
        ("avg_pool_3x3", 0),
        ("skip_connect", 2),
        ("avg_pool_3x3", 0),
    ],
    reduce_concat=range(2, 6),
)

SDARTS_RS_CIFAR10 = Genotype(
    normal=[
        ("sep_conv_3x3", 1),
        ("sep_conv_3x3", 0),
        ("sep_conv_5x5", 1),
        ("skip_connect", 0),
        ("sep_conv_3x3", 3),
        ("skip_connect", 1),
        ("sep_conv_3x3", 1),
        ("dil_conv_3x3", 2),
    ],
    normal_concat=range(2, 6),
    reduce=[
        ("max_pool_3x3", 0),
        ("sep_conv_3x3", 1),
        ("skip_connect", 2),
        ("max_pool_3x3", 0),
        ("dil_conv_5x5", 3),
        ("max_pool_3x3", 0),
        ("sep_conv_3x3", 2),
        ("sep_conv_5x5", 3),
    ],
    reduce_concat=range(2, 6),
)

SDARTS_ADV_CIFAR10 = Genotype(
    normal=[
        ("sep_conv_3x3", 0),
        ("sep_conv_3x3", 1),
        ("sep_conv_3x3", 1),
        ("skip_connect", 0),
        ("sep_conv_5x5", 0),
        ("dil_conv_3x3", 3),
        ("dil_conv_3x3", 4),
        ("skip_connect", 0),
    ],
    normal_concat=range(2, 6),
    reduce=[
        ("max_pool_3x3", 0),
        ("sep_conv_5x5", 1),
        ("skip_connect", 2),
        ("max_pool_3x3", 0),
        ("skip_connect", 3),
        ("skip_connect", 2),
        ("skip_connect", 2),
        ("sep_conv_5x5", 4),
    ],
    reduce_concat=range(2, 6),
)

DROPNAS = Genotype(
    normal=[
        ("skip_connect", 0),
        ("sep_conv_3x3", 1),
        ("sep_conv_3x3", 1),
        ("max_pool_3x3", 2),
        ("sep_conv_3x3", 1),
        ("sep_conv_5x5", 2),
        ("sep_conv_5x5", 0),
        ("sep_conv_5x5", 1),
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ("max_pool_3x3", 0),
        ("sep_conv_5x5", 1),
        ("dil_conv_5x5", 2),
        ("sep_conv_5x5", 1),
        ("dil_conv_5x5", 2),
        ("dil_conv_5x5", 3),
        ("dil_conv_5x5", 2),
        ("dil_conv_5x5", 4),
    ],
    reduce_concat=[2, 3, 4, 5],
)

GENAS_FLATNESS_CIFAR10 = parse_searched_cell(
    (3, 6, 0, 4, 4, 0, 6, 6, 0, 0, 0, 0, 4, 3, 5, 2, 0, 6, 4, 0, 3, 0, 6, 0, 0, 5, 0, 7)
)

GENAS_ANGLE_FLATNESS_CIFAR10 = parse_searched_cell(
    (5, 4, 4, 4, 0, 4, 5, 0, 0, 4, 0, 0, 0, 4, 7, 7, 2, 3, 0, 4, 1, 0, 0, 0, 0, 0, 5, 6)
)

GENAS_FLATNESS_CIFAR100 = parse_searched_cell(
    (3, 5, 0, 4, 4, 0, 0, 4, 5, 0, 4, 0, 0, 7, 6, 3, 2, 0, 5, 0, 5, 0, 7, 0, 1, 1, 0, 0)
)

GENAS_ANGLE_FLATNESS_CIFAR100 = parse_searched_cell(
    (3, 5, 1, 4, 0, 0, 4, 0, 4, 0, 4, 0, 0, 5, 6, 4, 0, 7, 2, 1, 3, 0, 0, 0, 4, 0, 0, 7)
)