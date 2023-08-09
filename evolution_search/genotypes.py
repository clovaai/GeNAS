"""
Code Adapted from https://github.com/megvii-model/RLNAS/blob/main/darts_search_space/cifar10/rlnas/evolution_search/genotypes.py
"""

from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

NASNet = Genotype(
  normal = [
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 0),
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 1),
    ('sep_conv_7x7', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('sep_conv_5x5', 0),
    ('skip_connect', 3),
    ('avg_pool_3x3', 2),
    ('sep_conv_3x3', 2),
    ('max_pool_3x3', 1),
  ],
  reduce_concat = [4, 5, 6],
)
    
AmoebaNet = Genotype(
  normal = [
    ('avg_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 2),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 3),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 1),
    ],
  normal_concat = [4, 5, 6],
  reduce = [
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 2),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('max_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('conv_7x1_1x7', 0),
    ('sep_conv_3x3', 5),
  ],
  reduce_concat = [3, 4, 6]
)

DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

def parse_searched_cell(normal_reduce_cell):
    '''
        normal_reduce_cell: list of normal + reduce cell. 
        e.g) [
            14 elements for normal cell edges where each element denote operation for each edge + 
            14 elements for reduce cell edges where each element denote operation for each edge
        ]
    '''
    assert len(normal_reduce_cell) == 28, "cell should contain normal + reduce edges (14 + 14 = 28)"
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

    return Genotype(normal=normal_cell_decoded, normal_concat=[2, 3, 4, 5], reduce=reduce_cell_decoded, reduce_concat=[2, 3, 4, 5])        

RLDARTS = parse_searched_cell((5, 4, 5, 0, 5, 0, 0, 5, 5, 0, 0, 0, 7, 4, 5, 4, 2, 0, 5, 0, 0, 4, 4, 0, 4, 4, 0, 0))
RLDARTS_GT = parse_searched_cell((5, 5, 4, 5, 0, 0, 4, 0, 4, 0, 0, 0, 4, 4, 1, 3, 3, 3, 0, 3, 2, 0, 0, 0, 4, 0, 0, 6))

DARTS = DARTS_V2

