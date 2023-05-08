# -*- coding: utf-8 -*-
# @Filename: bib_process
# @Date: 2022-06-17 14:25
# @Author: Leo Xu
# @Email: leoxc1571@163.com

import pandas as pd
import os
import codecs
import gzip
import sys
import csv

csv.field_size_limit(sys.maxsize)

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
RAW_DATASETS = 'dataset/bibsonomy_bm/tas'

def parse(path):
  g = open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

def get_helpful(x):
  x = x[1: -1]
  pos, total = x.split(', ')
  # pos, total = x[0], x[1]
  if int(total) == 0:
    return 0.5
  else:
    return int(pos)/int(total)


data = pd.read_csv('../dataset/bibsonomy_bm/tas', delimiter='\t', header=None, engine='python', quoting=csv.QUOTE_NONE)

print('done')