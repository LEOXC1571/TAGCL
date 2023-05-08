# !/usr/bin/env python
# -*- Coding: utf-8 -*-

# @Filename: run.py
# @Author: Leo Xu
# @Date: 2023/5/8 9:36
# @Email: leoxc1571@163.com
# Description:

import os
import argparse
from logging import getLogger

from recbole.config import Config
from recbole.data import data_preparation, save_split_dataloaders
from recbole.utils import init_logger, get_trainer, init_seed, set_color

from model import model_name_map
from data.dataset import TagBasedDataset
from data.utils import tag_data_preparation


def run(model=None, dataset=None, saved=False):
    current_path = os.path.dirname(os.path.realpath(__file__))
    overall_init_file = os.path.join(current_path, 'config/overall.yaml')
    model_init_file = os.path.join(current_path, 'config/model/' + model + '.yaml')
    dataset_init_file = os.path.join(current_path, 'config/dataset/' + dataset + '.yaml')
    config_file_list = [overall_init_file, model_init_file, dataset_init_file]

    model_class = model_name_map[model]
    config = Config(model=model_class, dataset=dataset, config_file_list=config_file_list)
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = TagBasedDataset(config)
    if config['save_dataset']:
        dataset.save()
    logger.info(dataset)

    if config['tag_neg_sampling']:
        train_data, valid_data, test_data = tag_data_preparation(config, dataset)
    else:
        train_data, valid_data, test_data = data_preparation(config, dataset)
    if config['save_dataloaders']:
        save_split_dataloaders(config, dataloaders=(train_data, valid_data, test_data))

    model = model_class(config, train_data.dataset).to(config['device'])
    logger.info(model)
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config['show_progress']
    )
    test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, action='store', help="model name")
    parser.add_argument("--dataset", type=str, action='store', help="dataset name")
    parser.add_argument("--saved", action='store_true', default=False)
    args, unknown = parser.parse_known_args()

    model_name = args.model
    dataset_name = args.dataset
    saved = args.saved
    run(model_name, dataset_name, saved)
