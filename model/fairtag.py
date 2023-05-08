# -*- coding: utf-8 -*-
# @Filename: fairtag.py
# @Date: 2022/10/23 16:35
# @Author: LEO XU
# @Email: leoxc1571@163.com

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
from model.layers import TransRTLoss
from model.abstract_recommender import TagRecommender


class FTAGCL(TagRecommender):
    input_type = InputType.PAIRWISE
    def __init__(self, config, dataset):
        super(FTAGCL, self).__init__(config, dataset)

        self.emb_size = config['embedding_size']
        self.cl_rate = config['ssl_weight']
        self.reg_weight = config['reg_weight']
        self.eps = config['eps']
        self.n_layers = config['n_layers']
        self.trans_weight = config['trans_weight']
        self.tag_neg_weight = config['tag_neg_weight']
        self.tau = config['tau']
        self.user_embedding = torch.nn.Embedding(self.n_users, self.emb_size)
        self.item_embedding = torch.nn.Embedding(self.n_items, self.emb_size)
        self.tag_embedding = torch.nn.Embedding(self.n_tags, self.emb_size)

        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.trans_score = TransRTLoss()

        self.user_tag_matrix = dataset.assign_matrix(self.USER_ID).astype(np.float32)
        self.item_tag_matrix = dataset.assign_matrix(self.ITEM_ID).astype(np.float32)
        self.ut_adj_matrix = self.get_norm_adj_mat(self.user_tag_matrix).to(self.device)
        self.it_adj_matrix = self.get_norm_adj_mat(self.item_tag_matrix).to(self.device)

        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e', 'restore_ut_e', 'restore_it_e']
        self.restore_user_e = None
        self.restore_item_e = None
        self.restore_ut_e = None
        self.restore_it_e = None

    def graph_construction(self):
        self.ut_graph = self.ut_adj_matrix
        self.it_graph = self.it_adj_matrix

    def get_uinorm_adj_mat(self):
        # build adj matrix
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_norm_adj_mat(self, matrix):
        n_field_row, n_field_col = matrix.shape  # add user item or user tag or item tag, adj is a squre
        A = sp.dok_matrix((n_field_row + n_field_col, n_field_row + n_field_col), dtype=np.float32)
        matrix_T = matrix.transpose()
        data_dict = dict(zip(zip(matrix.row, matrix.col + n_field_row), [1] * matrix.nnz))
        data_dict.update(dict(zip(zip(matrix_T.row + n_field_row, matrix_T.col), [1] * matrix.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)  # diag
        # add epsion to avoid divide by zero warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D  # la
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self):
        _user_embeddings = self.user_embedding.weight
        _item_embeddings = self.item_embedding.weight
        _tag_embeddings = self.tag_embedding.weight
        ut_ego_embeddings = torch.cat([_user_embeddings, _tag_embeddings], dim=0)
        it_ego_embeddings = torch.cat([_item_embeddings, _tag_embeddings], dim=0)
        return ut_ego_embeddings, it_ego_embeddings

    def forward(self, ut_graph, it_graph, perturbed=True):
        ut_ego, it_ego, = self.get_ego_embeddings()
        ut_emb_list = [ut_ego]
        it_emb_list = [it_ego]

        for i in range(self.n_layers):
            it_ego = torch.sparse.mm(it_graph, it_ego)
            if perturbed:
                ut_random_state = torch.randn(ut_ego.shape).to(self.device)
                it_random_state = torch.randn(it_ego.shape).to(self.device)
                ut_random_state = F.softmax(ut_random_state, dim=1)
                it_random_state = F.softmax(it_random_state, dim=1)
                ut_ego += ut_random_state * self.eps
                it_ego += it_random_state * self.eps
            ut_emb_list.append(ut_ego)
            it_emb_list.append(it_ego)

        ut_emb_list = torch.stack(ut_emb_list, dim=1)
        ut_emb_list = torch.mean(ut_emb_list, dim=1, keepdim=False)
        it_emb_list = torch.stack(it_emb_list, dim=1)
        it_emb_list = torch.mean(it_emb_list, dim=1, keepdim=False)

        u_ut_emb, t_ut_emb = torch.split(ut_emb_list, [self.n_users, self.n_tags], dim=0)
        i_it_emb, t_it_emb = torch.split(it_emb_list, [self.n_items, self.n_tags], dim=0)
        return u_ut_emb, i_it_emb, t_ut_emb, t_it_emb

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None \
                or self.restore_item_e is not None \
                or self.restore_ut_e is not None \
                or self.restore_it_e is not None:
            self.restore_user_e, self.restore_item_e, self.restore_ut_e, self.restore_it_e = None, None, None, None

        user_list = interaction[self.USER_ID]
        pos_item_list = interaction[self.ITEM_ID]
        neg_item_list = interaction[self.NEG_ITEM_ID]
        tag_list = interaction[self.TAG_ID]
        neg_tag_list = interaction[self.NEG_TAG_ID]

        u_ut_emb, i_it_emb, t_ut_emb, t_it_emb= self.forward(self.ut_adj_matrix, self.it_adj_matrix, perturbed=False)
        u_p1, i_p1, tu_p1, ti_p1 = self.forward(self.ut_graph, self.it_graph, perturbed=True)
        u_p2, i_p2, tu_p2, ti_p2 = self.forward(self.ut_graph, self.it_graph, perturbed=True)


        trans_loss = self.trans_loss(u_ut_emb[user_list], i_it_emb[pos_item_list],
                                     t_ut_emb[tag_list], t_it_emb[tag_list])
        ti_bpr_loss = self.tag_neg_weight * self.calc_itag_bpr_loss(i_it_emb, t_it_emb,
                                                                    pos_item_list, neg_item_list,tag_list, neg_tag_list)
        ut_bpr_loss = 0
        bpr_loss = self.calc_bpr_loss(u_ut_emb, i_it_emb, user_list, pos_item_list, neg_item_list)

        total_loss = bpr_loss + \
                     ut_bpr_loss + ti_bpr_loss + \
                     self.calc_ssl_loss(user_list, tag_list, u_p1, u_p2, tu_p1, tu_p2) + \
                     self.calc_ssl_loss(pos_item_list, tag_list, i_p1, i_p2, ti_p1, ti_p2) + \
                     self.trans_weight * trans_loss
        return total_loss

    def calc_itag_bpr_loss(self, i_emb, tag_emb, pos_i_list, neg_i_list, tag_list, neg_tag_list):
        pi_e = i_emb[pos_i_list]
        ni_e = i_emb[neg_i_list]
        pt_e = tag_emb[tag_list]
        nt_e = tag_emb[neg_tag_list]
        p_scores = torch.mul(pi_e, pt_e).sum(dim=1)
        n_scores = torch.mul(ni_e, nt_e).sum(dim=1)

        bpr_loss = torch.sum(-F.logsigmoid(p_scores - n_scores))
        return bpr_loss

    def calc_bpr_loss(self, user_emb, item_emb, user_list, pos_item_list, neg_item_list):
        u_e = user_emb[user_list]
        pi_e = item_emb[pos_item_list]
        ni_e = item_emb[neg_item_list]

        p_scores = torch.mul(u_e, pi_e).sum(dim=1)
        n_scores = torch.mul(u_e, ni_e).sum(dim=1)

        l1 = torch.sum(-F.logsigmoid(p_scores - n_scores))

        u_e_p = self.user_embedding(user_list)
        pi_e_p = self.item_embedding(pos_item_list)
        ni_e_p = self.item_embedding(neg_item_list)

        l2 = self.reg_loss(u_e_p, pi_e_p, ni_e_p)
        return l1 + l2 * self.reg_weight

    def calc_ssl_loss(self, user_list, pos_item_list, user_1, user_2, item_1, item_2):
        u_emb1 = F.normalize(user_1[user_list], dim=1)
        u_emb2 = F.normalize(user_2[user_list], dim=1)
        i_emb1 = F.normalize(item_1[pos_item_list], dim=1)
        i_emb2 = F.normalize(item_2[pos_item_list], dim=1)
        alluser = F.normalize(user_2, dim=1)
        allitem = F.normalize(item_2, dim=1)

        v1 = torch.sum(u_emb1 * u_emb2, dim=1)
        v2 = u_emb1.matmul(alluser.T)
        v1 = torch.exp(v1 / self.tau)
        v2 = torch.sum(torch.exp(v2 / self.tau), dim=1)
        ssl_user = -torch.sum(torch.log(v1 / v2))

        v3 = torch.sum(i_emb1 * i_emb2, dim=1)
        v4 = i_emb1.matmul(allitem.T)
        v3 = torch.exp(v3 / self.tau)
        v4 = torch.sum(torch.exp(v4 / self.tau), dim=1)
        ssl_item = -torch.sum(torch.log(v3 / v4))
        return (ssl_user + ssl_item) * self.cl_rate

    def trans_loss(self, user_emb, item_emb, utag_emb, itag_emb):
        trans = torch.norm(user_emb + (utag_emb - itag_emb) - item_emb, p=2)
        return trans

    def tag_trans_loss(self, user_emb, item_emb, neg_item_emb, utag_emb, itag_emb):
        trans_pos = torch.norm(user_emb + utag_emb + itag_emb - item_emb, p=2, keepdim=True)
        trans_neg = torch.norm(user_emb + utag_emb + itag_emb - neg_item_emb, p=2, keepdim=True)
        t2 = -F.logsigmoid(trans_neg - trans_pos)
        # t2 = -torch.log(t1)
        return t2

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        tag = interaction[self.TAG_ID]

        user_all_embeddings, item_all_embeddings, t_ut_emb, t_it_emb = self.forward(self.ut_adj_matrix,
                                                                                    self.it_adj_matrix, perturbed=False)

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        u_norm = torch.norm(u_embeddings, p=2, dim=1).reshape(-1, 1)
        i_norm = torch.norm(i_embeddings, p=2, dim=1).reshape(-1, 1)

        u_e = u_embeddings / u_norm
        i_e = i_embeddings / i_norm
        scores = torch.mul(u_e, i_e).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e, \
            self.restore_ut_e, self.restore_it_e = self.forward(self.ut_adj_matrix, self.it_adj_matrix, perturbed=False)
        user = self.restore_user_e[interaction[self.USER_ID]]
        return user.matmul(self.restore_item_e.T)

    def train(self, mode: bool = True):
        T = super().train(mode=mode)
        if mode:
            self.graph_construction()
        return T
