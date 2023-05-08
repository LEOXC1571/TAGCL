import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


def all_neighbor_sampler(X):
    matrix, max_deg = X
    matrix = matrix.tocsr()
    # max_deg = int(max(matrix.getnnz(1)))
    data = np.zeros((matrix.shape[0], max_deg), dtype=np.int)
    weight = np.zeros((matrix.shape[0], max_deg), dtype=np.int)
    for i in range(matrix.shape[0]):
        nonzeroId = matrix[i].nonzero()[1]
        x = len(nonzeroId)
        if x == 0:
            continue
        elif x < max_deg:
            sampleId = np.random.choice(nonzeroId, max_deg)
        else:
            sampleId = np.random.choice(nonzeroId, max_deg, replace=False)

        data[i] = sampleId + 1
        weight[i] = matrix[i].toarray()[0, sampleId]
    return [data, weight]


def neighbor_sampler(X):

    adj_w, k, device = X
    indexs = np.arange(adj_w[0].shape[1])
    np.random.shuffle(indexs)
    nei = [x[:, :k] for x in adj_w]
    # nei = torch.tensor(nei, dtype=torch.long, device=device)
    return nei


class Attention1(nn.Module):
    def __init__(self, in_features: int, atten_dim: int, dim_w: int):
        super().__init__()
        self.W_1 = nn.Parameter(torch.Tensor(in_features + dim_w, atten_dim))
        self.W_2 = nn.Parameter(torch.Tensor(in_features, atten_dim))
        self.b = nn.Parameter(torch.Tensor(1, atten_dim))
        self.v = nn.Parameter(torch.Tensor(1, atten_dim))
        self.act = nn.ReLU()

    def forward(self, ev, ej, ew, v_jw):
        zeroj = torch.zeros((1, ej.shape[1]), dtype=ej.dtype, device=ej.device)
        zerow = torch.zeros((1, ew.shape[1]), dtype=ew.dtype, device=ew.device)
        ej = torch.cat([zeroj, ej])
        ew = torch.cat([zerow, ew])
        v_j, v_w = v_jw
        k = v_j.shape[1]

        eNj = ej[v_j]
        eNw = ew[v_w]
        eNv = ev.unsqueeze(1).repeat(1, k, 1)

        eN_vw = torch.cat([eNv, eNw], dim=-1)
        av_j = torch.matmul(eN_vw, self.W_1) + torch.matmul(eNj, self.W_2) + self.b
        x = torch.matmul(self.act(av_j), self.v.T)
        a = torch.softmax(x, dim=1)
        eN = torch.sum(a * eNj, dim=1)
        return eN


class TGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, atten_dim, weight_dim, num_bit_conv, num_vector_conv):
        super(TGCNLayer, self).__init__()
        self._in_dim = in_feat
        self._num_vector_conv = num_vector_conv
        self._num_bit_conv = num_bit_conv

        self.atten1 = nn.ModuleDict()
        type_name = ["user", "item", "tag"]
        for type in type_name:
            self.atten1.update({type: Attention1(in_feat, atten_dim, weight_dim)})

        # attention2
        self.U = nn.Parameter(torch.Tensor(in_feat, atten_dim))
        self.q = nn.Parameter(torch.Tensor(1, atten_dim))
        self.p = nn.Parameter(torch.Tensor(1, atten_dim))

        self.conv = self._conv_layer()
        # fusion
        in_k = self._num_bit_conv * in_feat + self._num_vector_conv * (3 + 2 + 1)

        self.Wf = nn.Parameter(torch.Tensor(in_k, out_feat))
        self.bf = nn.Parameter(torch.Tensor(1, out_feat))
        self.act = nn.ReLU()

    def _conv_layer(self):
        vector_dict = nn.ModuleDict()
        for j in range(1, 4):
            vector_dict.update({
                f"conv_{j}": nn.Conv2d(1, self._num_vector_conv, kernel_size=(j, self._in_dim), bias=False),
            })
        conv_dict = nn.ModuleDict({
            "bit_level": nn.Conv2d(1, self._num_bit_conv, kernel_size=(3, 1), bias=False),
            "vec_level": vector_dict,
        })
        return conv_dict

    def _atten2(self, u, i, t):
        uit = torch.stack([u, i, t], dim=1)
        x = torch.matmul(uit, self.U) + self.q
        x = torch.matmul(self.act(x), self.p.T)
        b = torch.softmax(x, dim=1)
        x = b * uit
        return x

    def _conv(self, eN):
        x = eN.unsqueeze(dim=1)
        bit_e = self.conv["bit_level"](x)
        bit_e = self.act(bit_e)
        bit_e = bit_e.squeeze().reshape(bit_e.shape[0], -1)
        vec_e = []

        for model in self.conv["vec_level"].values():
            y = model(x)
            y = self.act(y)
            y = y.squeeze(dim=-1)
            vec_e.append(y.reshape(y.shape[0], -1))
        vector_e = torch.cat(vec_e, dim=-1)

        y = torch.cat([bit_e, vector_e], dim=1)
        return y

    def _fusion(self, x):
        x = torch.matmul(x, self.Wf) + self.bf
        x = self.act(x)
        return x

    def forward(self, eu, ei, et, ew, u_iw, u_tw, i_uw, i_tw, t_uw, t_iw):
        eu_uN = eu
        eu_iN = self.atten1["item"].forward(eu, ei, ew, u_iw)
        eu_tN = self.atten1["tag"].forward(eu, et, ew, u_tw)
        ei_iN = ei
        ei_uN = self.atten1["user"].forward(ei, eu, ew, i_uw)
        ei_tN = self.atten1["tag"].forward(ei, et, ew, i_tw)
        et_tN = et
        et_uN = self.atten1["user"].forward(et, eu, ew, t_uw)
        et_iN = self.atten1["item"].forward(et, ei, ew, t_iw)

        euN = self._atten2(eu_uN, eu_iN, eu_tN)
        eiN = self._atten2(ei_uN, ei_iN, ei_tN)
        etN = self._atten2(et_uN, et_iN, et_tN)
        eu_c = self._conv(euN)
        ei_c = self._conv(eiN)
        et_c = self._conv(etN)
        eu_k = self._fusion(eu_c)
        ei_k = self._fusion(ei_c)
        et_k = self._fusion(et_c)
        return eu_k, ei_k, et_k


class TGCN(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        # load dataset info
        self.TAG_ID = config['TAG_ID_FIELD']
        self.n_tags = dataset.num(self.TAG_ID)
        super(TGCN, self).__init__(config, dataset)
        # load dataset info graph
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.user_tag_matrix = dataset.assign_matrix(self.USER_ID).astype(np.float32)
        self.item_tag_matrix = dataset.assign_matrix(self.ITEM_ID).astype(np.float32)
        self.num_weight = int(max(self.interaction_matrix.max(),
                                  self.user_tag_matrix.max(),
                                  self.item_tag_matrix.max()))

        # load parameters info
        self.latent_dim = config['embedding_size']
        self.weight_dim = config['weight_dim']
        self.hidden_size_list = config['hidden_size_list']
        self.hidden_size_list = [self.latent_dim] + self.hidden_size_list
        self.atten_dim = config['atten_dim']
        self.num_bit_conv = config['num_bit_conv']
        self.num_vec_conv = config['num_vec_conv']
        self.message_drop_list = config['message_drop_list']
        self.device = config['device']
        self.neighbor_k = config['neighbor_k']
        self.reg_weight = config['reg_weight']
        self.transtag_reg = config['transtag_reg']
        self.loss_func = config['mul_loss_func']

        self.user_embedding = nn.Embedding(self.n_users, self.latent_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.latent_dim)
        self.tag_embedding = nn.Embedding(self.n_tags, self.latent_dim)
        self.weight_embedding = nn.Embedding(self.num_weight, self.weight_dim)

        self.GNNlayers = torch.nn.ModuleList()
        for idx, (input_size, output_size) in enumerate(zip(self.hidden_size_list[:-1], self.hidden_size_list[1:])):
            self.GNNlayers.append(
                TGCNLayer(input_size, output_size,
                          self.atten_dim, self.weight_dim, self.num_bit_conv, self.num_vec_conv)
            )

        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None
        self.restore_tag_e = None

        # generate intermediate data
        self.all_sample = self.get_all_neighbor()

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e', 'restore_tag_e']

    def get_all_neighbor(self):

        matrix = [self.interaction_matrix, self.user_tag_matrix, self.interaction_matrix.transpose(),
                  self.item_tag_matrix, self.user_tag_matrix.transpose(), self.item_tag_matrix.transpose()]

        max_deg = [max(adj.getnnz(1)) for adj in matrix]
        # X = zip(matrix, max_deg)
        #
        # pool = multiprocessing.Pool(6)  # cpu
        # results = pool.map(all_neighbor_sampler, X)
        # pool.close()
        #
        # return results

        neighbor_list = []
        for mat, deg in zip(matrix, max_deg):
            mat = mat.tocsr()
            data = np.zeros((mat.shape[0], deg), dtype=np.int)
            weight = np.zeros((mat.shape[0], deg), dtype=np.int)
            for i in range(mat.shape[0]):
                nonzeroId = mat[i].nonzero()[1]
                x = len(nonzeroId)
                if x == 0:
                    continue
                elif x < deg:
                    sampleId = np.random.choice(nonzeroId, deg)
                else:
                    sampleId = np.random.choice(nonzeroId, deg, replace=False)
                data[i] = sampleId + 1
                weight[i] = mat[i].toarray()[0, sampleId]
            neighbor_list.append([data, weight])

        return neighbor_list

    def sample(self):
        results = []
        for adj_w in self.all_sample:
            indexs = np.arange(adj_w[0].shape[1])
            np.random.shuffle(indexs)
            nei = [x[:, :self.neighbor_k] for x in adj_w]
            nei = torch.tensor(np.array(nei), dtype=torch.long, device=self.device)
            results.append(nei)
        return results

    def get_ego_embeddings(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        tag_embeddings = self.tag_embedding.weight
        weight_embeddings = self.weight_embedding.weight
        return user_embeddings, item_embeddings, tag_embeddings, weight_embeddings

    def forward(self):
        eu, ei, et, ew = self.get_ego_embeddings()
        u_embeddings_list = [eu]
        i_embeddings_list = [ei]
        t_embeddings_list = [et]

        for i, layer in enumerate(self.GNNlayers):
            neighbor = self.sample()
            u_iw, u_tw, i_uw, i_tw, t_uw, t_iw = neighbor
            eu, ei, et = layer(eu, ei, et, ew, u_iw, u_tw, i_uw, i_tw, t_uw, t_iw)
            del u_iw, u_tw, i_uw, i_tw, t_uw, t_iw
            torch.cuda.empty_cache()
            eu = F.dropout(eu, p=self.message_drop_list[i], training=self.training)
            ei = F.dropout(ei, p=self.message_drop_list[i], training=self.training)
            et = F.dropout(et, p=self.message_drop_list[i], training=self.training)
            eu_norm = F.normalize(eu, p=2, dim=1)
            ei_norm = F.normalize(ei, p=2, dim=1)
            et_norm = F.normalize(et, p=2, dim=1)
            u_embeddings_list.append(eu_norm)
            i_embeddings_list.append(ei_norm)
            t_embeddings_list.append(et_norm)
        u_embeddings_list = torch.cat(u_embeddings_list, dim=1)
        i_embeddings_list = torch.cat(i_embeddings_list, dim=1)
        t_embeddings_list = torch.cat(t_embeddings_list, dim=1)
        return u_embeddings_list, i_embeddings_list, t_embeddings_list

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None or self.restore_tag_e is not None:
            self.restore_user_e, self.restore_item_e, self.restore_tag_e = None, None, None

        user = interaction[self.USER_ID]
        tag = interaction[self.TAG_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        u_tgcn_embeddings, i_tgcn_embeddings, t_tgcn_embeddings = self.forward()
        u_embeddings = u_tgcn_embeddings[user]
        t_embeddings = t_tgcn_embeddings[tag]
        pos_embeddings = i_tgcn_embeddings[pos_item]
        neg_embeddings = i_tgcn_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate Reg Loss
        u_ego_embeddings = self.user_embedding(user)
        t_ego_embeddings = self.tag_embedding(tag)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(u_ego_embeddings, t_ego_embeddings, pos_ego_embeddings)
        loss = mf_loss + self.reg_weight * reg_loss
        return loss

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
