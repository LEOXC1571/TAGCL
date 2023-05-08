# !/usr/bin/env python
# -*- Coding: utf-8 -*-

# @Filename: tune.py
# @Author: Leo Xu
# @Date: 2023/5/8 9:35
# @Email: leoxc1571@163.com
# Description:

import os
import logging
import argparse

from recbole.trainer import HyperTuning
from recbole.utils import ensure_dir
from recbole.config import Config
from recbole.data import data_preparation
from recbole.utils import get_trainer, init_seed

from model import model_name_map
from data.dataset import TagBasedDataset
from data.utils import tag_data_preparation


def objective_run(config_dict=None, config_file_list=None, comments='may', saved=False):
    model_config_str, dataset_config_str = config_file_list[1:]
    model_name = model_config_str.split('/')[-1].split('.')[0]
    dataset = dataset_config_str.split('/')[-1].split('.')[0]
    model_class = model_name_map[model_name]
    config = Config(model=model_class, dataset=dataset, config_dict=config_dict, config_file_list=config_file_list)
    config['checkpoint_dir'] = config['checkpoint_dir'] + '-' + comments
    init_seed(config['seed'], config['reproducibility'])
    logging.basicConfig(level=logging.ERROR)
    dataset = TagBasedDataset(config)
    if config['tag_neg_sampling']:
        train_data, valid_data, test_data = tag_data_preparation(config, dataset)
    else:
        train_data, valid_data, test_data = data_preparation(config, dataset)
    model = model_class(config, train_data.dataset).to(config['device'])
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=False, saved=saved)
    test_result = trainer.evaluate(test_data, load_best_model=saved)
    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


def tune(model=None, dataset=None, params_name=None, comments='may'):
    current_path = os.path.dirname(os.path.realpath(__file__))
    overall_init_file = os.path.join(current_path, 'config/overall.yaml')
    model_init_file = os.path.join(current_path, 'config/model/' + model + '.yaml')
    dataset_init_file = os.path.join(current_path, 'config/dataset/' + dataset + '.yaml')
    config_file_list = [overall_init_file, model_init_file, dataset_init_file]
    params_space = os.path.join(current_path, 'config/params/' + params_name + '.params')

    hp = HyperTuning(objective_run, algo='exhaustive',
                     params_file=params_space, fixed_config_file_list=config_file_list)
    hp.run()
    experiment_result_path = f'hyper_experiment_{comments}'
    ensure_dir(experiment_result_path)
    hp.export_result(output_file=f'{experiment_result_path}/{model}-{dataset}.result')
    print('best params: ', hp.best_params)
    print('best result: ')
    print(hp.params2result[hp.params2str(hp.best_params)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, action='store', help="model name")
    parser.add_argument("--dataset", type=str, action='store', help="dataset name")
    parser.add_argument("--params", type=str, default=None, help='parameters file')
    parser.add_argument("--comments", type=str, default=None, help='experiment comments')
    args, unknown = parser.parse_known_args()

    model_name = args.model
    dataset_name = args.dataset
    params_name = args.params
    comments = args.comments
    tune(model_name, dataset_name, params_name, comments)
