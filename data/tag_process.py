#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :tag_process.py
# @Time      :2021/11/4 下午7:50
# @Author    :Miracleyin
# @Mail      :miracleyin@live.com

import csv
import os
import time
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from data.base_dataset import BaseDataset


class HETRECProcess(BaseDataset):
    def __init__(self, input_path, output_path, dataset_name='movielens'):
        super(HETRECProcess, self).__init__(input_path, output_path)
        _least_k = {"delicious": 15, "lastfm": 5, "movielens": 5, "bibsonomy_bm": 30, "bibsonomy_bt": 30}
        _inter_file = {"delicious": 'user_taggedbookmarks-timestamps.dat',
                       "lastfm": 'user_taggedartists-timestamps.dat',
                       "movielens": 'user_taggedmovies-timestamps.dat',
                       "bibsonomy_bm": 'tas',
                       "bibsonomy_bt": 'tas'}
        self.dataset_name = dataset_name
        self._least_k = _least_k[dataset_name]

        self.inter_file = os.path.join(self.input_path, _inter_file[dataset_name])

        self.sep = "\t"

        self.output_inter_file, self.output_assign_file = self._get_output_files()

        if self.dataset_name == 'bibsonomy_bm' or self.dataset_name == 'bibsonomy_bt':
            self.inter_fields = {0: 'user_id:token',
                                 1: 'tag_id:token',
                                 2: 'item_id:token',
                                 3: 'timestamp:float'}  # useful col
            self.assign_fields = {0: 'user_id:token',
                                  1: 'tag_id:token',
                                  2: 'item_id:token',
                                  3: 'timestamp:float'}
        else:
            self.inter_fields = {0: 'user_id:token',
                                 1: 'item_id:token',
                                 2: 'tag_id:token'}  # useful col
            self.assign_fields = {0: 'user_id:token',
                                  1: 'item_id:token',
                                  2: 'tag_id:token'}

    def _get_output_files(self):
        output_inter_file = os.path.join(self.output_path, self.dataset_name + '.inter')
        output_assign_file = os.path.join(self.output_path, self.dataset_name + '.assign')
        return output_inter_file, output_assign_file

    def load_inter_data(self):
        if self.dataset_name == 'bibsonomy_bm' or self.dataset_name == 'bibsonomy_bt':
            origin_data = pd.read_csv(self.inter_file, delimiter='\t', header=None,
                                      engine='python', quoting=csv.QUOTE_NONE)
            origin_data.columns = ['user_id', 'tag_id', 'item_id', 'content_type', 'date']
            if self.dataset_name == 'bibsonomy_bm':
                origin_data = origin_data[origin_data['content_type'] == 1]
            elif self.dataset_name == 'bibsonomy_bt':
                origin_data = origin_data[origin_data['content_type'] == 2]
            origin_data['tag_id'] = LabelEncoder().fit_transform(origin_data['tag_id'])
            # origin_data['date'] = pd.to_datetime(origin_data['date'], format='%Y/%m/%d')
            origin_data['timestamp'] = origin_data['date'].apply(lambda x: time.mktime(
                time.strptime(x, '%Y-%m-%d %H:%M:%S')))
            origin_data.drop(['content_type', 'date'], axis=1, inplace=True)
            # origin_data = origin_data.iloc[:, :3]  # 取出 user_id item_id tag_id
            tag_data = origin_data['tag_id'].value_counts()
            del_tag = list(tag_data[origin_data['tag_id']] >= self._least_k)
            origin_data = origin_data[del_tag]
            origin_data.reset_index(drop=True, inplace=True)
            return origin_data

        origin_data = pd.read_csv(self.inter_file, delimiter=self.sep, engine='python')
        origin_data = origin_data.iloc[:, :3]  # 取出 user_id item_id tag_id
        tag_data = origin_data['tagID'].value_counts()
        del_tag = list(tag_data[origin_data['tagID']] >= self._least_k)
        origin_data = origin_data[del_tag]
        origin_data.reset_index(drop=True, inplace=True)
        return origin_data

    def load_assign_data(self):
        origin_data = pd.read_csv(self.inter_file, delimiter=self.sep, engine='python')
        origin_data = origin_data.iloc[:, :3]  # 取出 user_id item_id tag_id
        tag_data = origin_data['tagID'].value_counts()
        del_tag = list(tag_data[origin_data['tagID']] >= self._least_k)
        origin_data = origin_data[del_tag]
        origin_data.reset_index(drop=True, inplace=True)
        return origin_data

    def convert_assign(self):
        try:
            input_inter_data = self.load_assign_data()
            self.convert(input_inter_data, self.assign_fields, self.output_assign_file)
        except NotImplementedError:
            print('This dataset can\'t be converted to inter file\n')


if __name__ == '__main__':
    data = 'movielens'
    input_path = '../raw'
    output_path = f'../dataset/{data}'
    l = HETRECProcess(input_path, output_path, data)
    l.convert_inter()
