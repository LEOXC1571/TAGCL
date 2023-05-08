# !/usr/bin/env python
# -*- Coding: utf-8 -*-

# @Filename: abstract_recommender.py
# @Author: Leo Xu
# @Date: 2023/5/8 10:32
# @Email: leoxc1571@163.com
# Description:

from recbole.utils import ModelType
from recbole.model.abstract_recommender import GeneralRecommender


class TagRecommender(GeneralRecommender):
    type = ModelType.GENERAL

    def __init__(self, config, dataset):
        super(GeneralRecommender, self).__init__()

        # load dataset info
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID
        self.TAG_ID = config['TAG_ID_FIELD']
        self.NEG_TAG_ID = config['NEG_PREFIX'] + self.TAG_ID
        self.n_users = dataset.num(self.USER_ID)
        self.n_items = dataset.num(self.ITEM_ID)
        self.n_tags = dataset.num(self.TAG_ID)

        self.device = config['device']
