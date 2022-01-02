import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import random
import math
import copy
import gc
import tqdm
import glob
from konlpy.tag import Mecab
import torch
from torch import nn
from torch.utils.data import Dataset, dataloader

import warnings
# warnings.filterwarning(action = "ignore")

def seed_everything(seed :  int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] =  str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministc =  True
    torch.backends.cudnn.benchmark =  True

seed_everything(42)

DIR =  "/content/drive/MyDrive/235813_AI 기반 회의 녹취록 요약 경진대회_data"
TRAIN_SOURCE =  os.path.join(DIR, "train.json")
TEST_SOURCE = os.path.join(DIR, "test.json")

with open(TRAIN_SOURCE) as f:
    train_data =  json.loads(f.read())
with open(TEST_SOURCE) as f:
    test_data = json.loads(f.read())

train =  pd.DataFrame(columns= ["uid", "title", "region", "context", "summary"])
uid = 1000

for data in train_data:
    for agenda in data["context"].keys(): #안건 번호 
        context = ""
    for line in data["context"][agenda]: #각 안건의 한줄 한줄에 대해
        context += data["context"][agenda][line]
        context += " "

    train.loc[uid, "uid"] = uid
    train.loc[uid, "title"] = data["title"]
    train.loc[uid, "region"] = data["region"]
    train.loc[uid, "context"] = context[:-1]
    train.loc[uid, "summary"] =  data["label"][agenda]["summary"]
    uid += 1

test =  pd.DataFrame(columns= ["uid", "title","region","context"])
uid = 2000
for data in test_data:
    for agenda in data['context'].keys():
        context = ''
        for line in data['context'][agenda]:
            context += data['context'][agenda][line]
            context += ' '
        test.loc[uid, 'uid'] = uid
        test.loc[uid, 'title'] = data['title']
        test.loc[uid, 'region'] = data['region']
        test.loc[uid, 'context'] = context[:-1]
        uid += 1
train["total"] = train.title + " " + train.region + " " +train.context
test["total"] = test.title + " " + test.region + " " + test.context

train.head()