#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :dataset.py
# @Time      :2021/11/4 下午9:56
# @Author    :Miracleyin
# @Mail      :miracleyin@live.com

import os
import time

import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix
from collections import Counter, defaultdict
from recbole.data.dataset import Dataset
from recbole.data.interaction import Interaction
from recbole.utils.enum_type import FeatureType
from recbole.utils import FeatureSource, FeatureType, get_local_time, set_color
# from recbole.utils import FeatureType, get_local_time, set_color
# from utils.enum_type import FeatureSource


class TagBasedDataset(Dataset):
    """:class:`TagBasedDataset` is based on :`~recbole.data.dataset.dataset.Dataset`,
    and load_col 'tag_id' additionally

    tag assisgment [``user_id``, ``item_id``, ``tag_id``]

    Attributes:
        tid_field (str): The same as ``config['TAG_ID_FIELD']``

    """

    def __init__(self, config):
        super().__init__(config)

    def _get_field_from_config(self):
        super()._get_field_from_config()

        self.tid_field = self.config['TAG_ID_FIELD']
        self._check_field('tid_field')
        self.set_field_property(self.tid_field, FeatureType.TOKEN, FeatureSource.INTERACTION, 1)
        if self.tid_field is None:
            raise ValueError(
                'Tag need to be set at the same time or not set at the same time.'
            )
        self.logger.debug(set_color('tid_field', 'blue') + f': {self.tid_field}')

    def _data_filtering(self):
         super()._data_filtering()
         # self._filter_tag()

    # def _filter_tag(self):
    #     pass

    #def _load_data(self, token, dataset_path):
    #    super()._load_data(token, dataset_path)

    def _load_data(self, token, dataset_path):
        if not os.path.exists(dataset_path):
            self._download()
        self._load_inter_feat(token, dataset_path)
        self.user_feat = self._load_user_or_item_feat(token, dataset_path, FeatureSource.USER, 'uid_field')
        self.item_feat = self._load_user_or_item_feat(token, dataset_path, FeatureSource.ITEM, 'iid_field')
        self.tag_feat = None
        self._load_additional_feat(token, dataset_path)

    def __str__(self):
        info = [
            super().__str__()
        ]
        if self.tid_field:
            info.extend([
                set_color('The number of tags', 'blue') + f': {self.tag_num}',
                set_color('Average actions of tags', 'blue') + f': {self.avg_actions_of_tags}'
            ])
        return '\n'.join(info)

    # def _build_feat_name_list(self):
    #     feat_name_list = super()._build_feat_name_list()
    #     # feat_name_list = [
    #     #     feat_name for feat_name in ['inter_feat', 'user_feat', 'item_feat']
    #     #     if getattr(self, feat_name, None) is not None
    #     # ]
    #     # if self.config['additional_feat_suffix'] is not None:
    #     #     for suf in self.config['additional_feat_suffix']:
    #     #         if getattr(self, f'{suf}_feat', None) is not None:
    #     #             feat_name_list.append(f'{suf}_feat')
    #     if self.tid_field is not None:
    #         feat_name_list.append('tag_feat')
    #     return feat_name_list

    def _get_remap_list(self, field_list):
        remap_list = []
        for field in field_list:
            ftype = self.field2type[field]
            for feat in self.field2feats(field):
                remap_list.append((feat, field, ftype))
        return remap_list

    def _init_alias(self):
        """Add :attr:`alias_of_tag_id` and update :attr:`_rest_fields`.
        """
        super()._init_alias()
        self._set_alias('tag_id', [self.tid_field])

    def assign_matrix(self, field, form='coo', value_field=None):
        if not self.tid_field or not field:
            raise ValueError(f'dataset does not exist tid/{field} thus can not converted to sparse matrix.')
        return self._create_sparse_matrix(self.inter_feat, field, self.tid_field, form, value_field)

    def _create_sparse_matrix(self, df_feat, source_field, target_field, form='coo', value_field=None):
        src = df_feat[source_field]
        tgt = df_feat[target_field]
        if value_field is None:
            data = np.ones(len(df_feat))
        else:
            if value_field not in df_feat:
                raise ValueError(f'Value_field [{value_field}] should be one of `df_feat`\'s features.')
            data = df_feat[value_field]
        mat = coo_matrix((data, (src, tgt)), shape=(self.num(source_field), self.num(target_field)))

        if form == 'coo':
            return mat
        elif form == 'csr':
            return mat.tocsr()
        else:
            raise NotImplementedError(f'Sparse matrix format [{form}] has not been implemented.')

    def create_src_tgt_matrix(self, df_feat, source_field, target_field, is_weight=True):
        """Get sparse matrix that describe relations between two fields.

        Source and target should be token-like fields.

        Sparse matrix has shape (``self.num(source_field)``, ``self.num(target_field)``).

        For a row of <src, tgt>, ``matrix[src, tgt] = 1`` if ``value_field`` is ``None``,
        else ``matrix[src, tgt] = df_feat[value_field][src, tgt]``.

        Args:
            df_feat (Interaction): Feature where src and tgt exist.
            source_field (str): Source field
            target_field (str): Target field
            form (str, optional): Sparse matrix format. Defaults to ``coo``.
            value_field (str, optional): Data of sparse matrix, which should exist in ``df_feat``.
                Defaults to ``None``.

        Returns:
            scipy.sparse: Sparse matrix in form ``coo`` or ``csr``.
        """
        if not isinstance(df_feat, pd.DataFrame):
            try:
                df_feat = pd.DataFrame.from_dict(df_feat.interaction)
            except BaseException:
                raise ValueError(f'feat from is not supported.')
        df_feat = df_feat.groupby([source_field, target_field]).size()
        df_feat.name = 'weights'
        df_feat = df_feat.reset_index()
        src = df_feat[source_field]
        tgt = df_feat[target_field]
        if is_weight:
            data = df_feat['weights']
        else:
            data = np.ones(len(df_feat))
        mat = coo_matrix((data, (src, tgt)), shape=(self.num(source_field), self.num(target_field)))
        return mat
        # if form == 'coo':
        #     return mat
        # elif form == 'csr':
        #     return mat.tocsr()
        # else:
        #     raise NotImplementedError(f'Sparse matrix format [{form}] has not been implemented.')

    def inter_graph(self, form='dgl', value_field=None):

        if not self.uid_field or not self.iid_field:
            raise ValueError('dataset does not exist uid/iid, thus can not converted to sparse matrix.')
        return self._create_graph(self.inter_feat, self.uid_field, self.iid_field, form, value_field)

    @property
    def tag_num(self):
        """Get the number of different tokens of tags.

       Returns:
           int: Number of different tokens of tags.
       """
        return self.num(self.tid_field)

    @property
    def avg_actions_of_tags(self):
        """Get the average number of tags' interaction records.

        Returns:
             numpy.float64: Average number of tags' interaction records.
        """
        if isinstance(self.inter_feat, pd.DataFrame):
            return np.mean(self.inter_feat.groupby(self.tid_field).size())
        else:
            return np.mean(list(Counter(self.inter_feat[self.tid_field].numpy()).values()))

    def field2feats(self, field):
        if field not in self.field2source:
            raise ValueError(f'Field [{field}] not defined in dataset.')
        if field == self.uid_field:
            feats = [self.inter_feat]
            if self.user_feat is not None:
                feats.append(self.user_feat)
        elif field == self.iid_field:
            feats = [self.inter_feat]
            if self.item_feat is not None:
                feats.append(self.item_feat)
        elif field == self.tid_field:
            feats = [self.inter_feat]
            if self.tag_feat is not None:
                feats.append(self.tag_feat)
        else:
            source = self.field2source[field]
            if not isinstance(source, str):
                source = source.value
            feats = [getattr(self, f'{source}_feat')]
        return feats

