import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
from model.layers import TransRTLoss


class LAGCF(GeneralRecommender):
    input_type = InputType.PAIRWISE
    def __init__(self, config, dataset):
        # load dataset info
        self.TAG_ID = config['TAG_ID_FIELD']
        self.n_tags = dataset.num(self.TAG_ID)
        super(LAGCF, self).__init__(config, dataset)
        # load dataset info graph
        # self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)  # user item mat
        self.user_tag_matrix = dataset.assign_matrix(self.USER_ID).astype(np.float32)
        self.item_tag_matrix = dataset.assign_matrix(self.ITEM_ID).astype(np.float32)

        # load parameters info
        self.latent_dim = config['embedding_size']  # int type:the embedding size of lightGCN
        self.n_layers = config['n_layers']  # int type:the layer num of lightGCN
        self.reg_weight = config['reg_weight']  # float32 type: the weight dec·ay for l2 normalization
        self.ut_weight = config['ut_weight']
        self.it_weight = config['it_weight']
        self.require_pow = config['require_pow']

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.latent_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.latent_dim)
        self.tag_embedding = nn.Embedding(self.n_tags, self.latent_dim)
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.trans_score = TransRTLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None
        self.restore_tag_e = None

        # generate intermediate data
        self.norm_ut_adj_matrix = self.get_norm_adj_mat(self.user_tag_matrix).to(self.device)
        self.norm_it_adj_matrix = self.get_norm_adj_mat(self.item_tag_matrix).to(self.device)

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e', 'restore_tag_e']

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

    def get_ego_embeddings(self):  # returen ut or it embedding
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        tag_embeddings = self.tag_embedding.weight
        # inter_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        assign_ut_embeddings = torch.cat([user_embeddings, tag_embeddings], dim=0)
        assign_it_embeddings = torch.cat([item_embeddings, tag_embeddings], dim=0)

        return assign_ut_embeddings, assign_it_embeddings

    def forward(self):
        assign_ut_embeddings, assign_it_embeddings = self.get_ego_embeddings()
        ut_embeddings_list = [assign_ut_embeddings]
        it_embeddings_list = [assign_it_embeddings]

        for layer_idx in range(self.n_layers):
            assign_ut_embeddings = torch.sparse.mm(self.norm_ut_adj_matrix, assign_ut_embeddings)
            assign_it_embeddings = torch.sparse.mm(self.norm_it_adj_matrix, assign_it_embeddings)
            ut_embeddings_list += [assign_ut_embeddings]
            it_embeddings_list += [assign_it_embeddings]

        lgcn_assign_ut_embeddings = torch.stack(ut_embeddings_list, dim=1)
        lgcn_assign_ut_embeddings = torch.mean(lgcn_assign_ut_embeddings, dim=1)
        u_assign_ut_embeddings, t_assign_ut_embeddings = torch.split(lgcn_assign_ut_embeddings,
                                                                     [self.n_users, self.n_tags])
        lgcn_assign_it_embeddings = torch.stack(it_embeddings_list, dim=1)
        lgcn_assign_it_embeddings = torch.mean(lgcn_assign_it_embeddings, dim=1)
        i_assign_it_embeddings, t_assign_it_embeddings = torch.split(lgcn_assign_it_embeddings,
                                                                     [self.n_items, self.n_tags])

        return u_assign_ut_embeddings, i_assign_it_embeddings, t_assign_ut_embeddings, t_assign_it_embeddings

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None or self.restore_tag_e is not None:
            self.restore_user_e, self.restore_item_e, self.restore_tag_e = None, None, None

        user = interaction[self.USER_ID]
        tag = interaction[self.TAG_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        u_assign_embeddings, i_assign_embeddings, t_assign_ut_embeddings, t_assign_it_embeddings = self.forward()

        u_embeddings = u_assign_embeddings[user]
        t_ut_embeddings = t_assign_ut_embeddings[tag]
        t_it_embeddings = t_assign_it_embeddings[tag]
        pos_embeddings = i_assign_embeddings[pos_item]
        neg_embeddings = i_assign_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate Reg Loss
        u_ego_embeddings = self.user_embedding(user)
        t_ego_embeddings = self.tag_embedding(tag)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(u_ego_embeddings, t_ego_embeddings, pos_ego_embeddings,
                                 require_pow=self.require_pow)
        trans_loss_u = self.trans_score(u_ego_embeddings, t_ut_embeddings, pos_ego_embeddings)
        trans_loss_i = self.trans_score(u_ego_embeddings, t_it_embeddings, pos_ego_embeddings)

        loss = mf_loss + self.reg_weight * reg_loss + self.ut_weight * trans_loss_u + self.it_weight * trans_loss_i
        return loss

    def _get_kg_embedding(self, user, tag, pos_i, neg_i):
        user_e = self.user_embedding(user)
        tag_e = self.tag_embedding(tag)
        pos_e = self.item_embedding(pos_i)
        neg_e = self.item_embedding(neg_i)
        return user_e, tag_e, pos_e, neg_e

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings, *_ = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e, *_ = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)
