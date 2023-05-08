#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :bprt.params.py
# @Time      :2021/11/25 下午9:30
# @Author    :Miracleyin
# @Mail      :miracleyin@live.com

import numpy as np
import torch
import torch.nn as nn

from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType
from model.abstract_recommender import TagRecommender


class ExtendedBPR(TagRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(ExtendedBPR, self).__init__(config, dataset)

        self.TAG_ID = config['TAG_ID_FIELD']
        # load parameters info
        self.embedding_size = config['embedding_size']
        self.lammda = config['lammda']
        self.beta = config['beta']
        self.gamma = config['gamma']

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.tag_embedding = nn.Embedding(self.n_tags, self.embedding_size)
        self.loss = BPRLoss()

        # get user-tag weight metrix A item-tag weight metrix B
        self.user_tag_value_matrix = dataset.create_src_tgt_matrix(dataset.inter_feat, self.USER_ID, self.TAG_ID)
        self.item_tag_value_matrix = dataset.create_src_tgt_matrix(dataset.inter_feat, self.ITEM_ID, self.TAG_ID)
        self.user_tag_value_matrix = self._get_weights_mat(self.user_tag_value_matrix).to_dense()
        self.item_tag_value_matrix = self._get_weights_mat(self.item_tag_value_matrix).to_dense()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def get_user_embedding(self, user):
        return self.user_embedding(user)

    def get_item_embedding(self, item):
        return self.item_embedding(item)

    def get_tag_embedding(self, tag):
        return self.tag_embedding(tag)

    def get_user_tag_weight(self, tag):
        pass

    def _get_weights_mat(self, weight_mat):
        mat_data = torch.FloatTensor(weight_mat.data)
        mat_indices = torch.FloatTensor(np.vstack((weight_mat.row, weight_mat.col)))
        return torch.sparse_coo_tensor(mat_indices, mat_data, weight_mat.shape).to(self.device)

    def forward(self, user, item, tag):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        tag_e = self.get_tag_embedding(tag)
        return user_e, item_e, tag_e

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        tag = interaction[self.TAG_ID]

        user_e, pos_e, tag_e = self.forward(user, pos_item, tag)
        neg_e = self.get_item_embedding(neg_item)
        pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(user_e, neg_e).sum(dim=1)
        reg_ui = self._regularization_uit_para(user_e, pos_e, tag_e)
        reg_e2i = self._regularization_e2i_map(user, pos_item)
        reg_uneighbor = self._regularization_user_neighbor(user)
        loss = self.loss(pos_item_score, neg_item_score)
        return loss + reg_ui + reg_e2i + reg_uneighbor

    def _regularization_uit_para(self, user_e, pos_e, tag_e):
        reg = self.lammda * (torch.norm(user_e) ** 2 + torch.norm(pos_e) ** 2 + torch.norm(tag_e) ** 2) / 2
        return reg / user_e.shape[0]

    def _regularization_e2i_map(self, user, pos):
        ut_mat, it_mat = self.user_tag_value_matrix, self.item_tag_value_matrix
        A_u, B_i = ut_mat[user], it_mat[pos]
        user_weights_e = torch.mm(A_u, self.tag_embedding.weight) - self.get_user_embedding(user)
        item_weights_e = torch.mm(B_i, self.tag_embedding.weight) - self.get_item_embedding(pos)
        reg = self.beta * (torch.norm(user_weights_e) ** 2 + torch.norm(item_weights_e) ** 2) / 2
        return reg / user.shape[0]

    def _full_sort_similarity(self, user, k=1+10):
        user_e = self.get_user_embedding(user)
        all_user_e = self.user_embedding.weight
        src_tgt_sim = torch.matmul(user_e, all_user_e.transpose(0, 1))
        src_user_e = torch.norm(user_e, dim=1).unsqueeze(1)
        tgt_user_e = torch.norm(all_user_e, dim=1).unsqueeze(0)
        normlize = torch.mm(src_user_e, tgt_user_e)
        score = src_tgt_sim / normlize
        sim_score, topk_idx = torch.topk(score, k=k)
        sim_score, topk_idx = sim_score[:, 1:], topk_idx[:, 1:]
        sim_user_e = all_user_e[topk_idx]

        return sim_score, topk_idx

    def _regularization_user_neighbor(self, user):
        sim_score, topk_idx = self._full_sort_similarity(user)
        user_e = self.get_user_embedding(user)
        all_user_e = self.user_embedding.weight
        sim_user_e = all_user_e[topk_idx]
        nor_sim_score = torch.sum(sim_score, dim=1).unsqueeze(1)
        sim_user_score_e = torch.mul(sim_score.unsqueeze(2), sim_user_e)
        sim_user_score_e = torch.sum(sim_user_score_e, dim=1).squeeze()
        sim_user_weight = sim_user_score_e / nor_sim_score
        reg = self.gamma * torch.norm(user_e - sim_user_weight) / 2
        return reg / user.shape[0]

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        tag = interaction[self.TAG_ID]
        user_e, item_e, _ = self.forward(user, item, tag)
        return torch.mul(user_e, item_e).sum(dim=1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)

    def _similarity(self, src_user_e, tgt_user_e):
        return src_user_e * tgt_user_e / (torch.norm(src_user_e) * torch.norm(tgt_user_e))
