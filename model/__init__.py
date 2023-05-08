#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :__init__.py
# @Time      :2021/11/4 下午7:12
# @Author    :Miracleyin
# @Mail      :miracleyin@live.com

from .bprt import ExtendedBPR
from .lightgcn import LightGCN
from .lagcf import LAGCF
from .tgcn import TGCN
from .simgcl import SimGCL
from .fairtag import FTAGCL
# from .tsgl import TSGL
from recbole.model.general_recommender import Pop

recbole_models = { # for temp using recbole inner model
    'BPR',
    'Pop'
}

model_name_map = {
    'Pop': Pop,
    'BPR-T': ExtendedBPR,
    'LGCN': LightGCN,
    'LFGCF': LAGCF,
    'TGCN': TGCN,
    'SimGCL': SimGCL,
    'TAGCL': FTAGCL
}
