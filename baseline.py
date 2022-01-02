from typing import Text
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

DIR =  "/Users/seojiwon/Downloads/sum_open"
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

encoder_len = 500
decoder_len = 50
max_vocab_size = 20000
batch_size = 32
num_layers = 6
d_model = 512
dff = 2048
num_heads = 8
dropout_rate = 0.1
epochs = 20
learning_rate = 1e-4
device = torch.device("cuda:0")

df_train = train.iloc[:-200]
df_val = train.iloc[-200:]

class Mecab_Tokenizer():
    def __init__(self, max_length, mode, max_vocab_size=-1):
        self.text_tokenizer = Mecab()
        self.mode = mode
        self.txt2idx = {'pad_':0, 'unk_':1}
        self.idx2txt = {0:'pad_', 1:'unk_'}
        self.max_length = max_length
        self.word_count = {}
        self.max_vocab_size = max_vocab_size
        
        # 띄어쓰기를 찾기 위한 태그 목록
        self.font_blank_tag = [
            '', 'EC', 'EC+JKO', 'EF', 'EP+EC', 'EP+EP+EC', 'EP+ETM', 'EP+ETN+JKO', 'ETM', 'ETN', 'ETN+JKO', 'ETN+JX', 'IC', 'JC', 'JKB', 'JKB+JX', 'JKO',
            'JKQ', 'JKS', 'JX', 'MAG', 'MAG+JX', 'MAG+XSV+EP+EC', 'MAJ','MM', 'MM+EC', 'NNB', 'NNB+JKB', 'NNB+JKO', 'NNB+VCP+EC', 'NNBC', 'NNG', 'NNG+JX+JKO',
            'NNG+VCP+EC', 'NNP', 'NNP+JX', 'NP', 'NP+JKO', 'NP+JKS', 'NP+JX', 'NP+VCP+EC', 'NR', 'SC', 'SF', 'SL', 'SN', 'SSC', 'SSO', 'SY', 'UNKNOWN',
            'VA+EC', 'VA+EC+VX+ETM', 'VA+ETM', 'VA+ETN+JKB+JX', 'VCN+EC', 'VCN+ETM', 'VCP', 'VCP+EC', 'VCP+EP+EC', 'VCP+EP+ETM', 'VCP+ETM', 'VCP+ETN',
            'VV+EC', 'VV+EC+JX', 'VV+EC+VX+EC', 'VV+EC+VX+ETM', 'VV+EP+EC', 'VV+EP+ETM', 'VV+ETM', 'VV+ETN', 'VX+EC', 'VX+EC+VX+EP+EC', 'VX+EP+ETM',
            'VX+ETM', 'XPN', 'XR', 'XSA+EC', 'XSA+EC+VX+ETM', 'XSA+ETM', 'XSN', 'XSV+EC', 'XSV+EP+EC', 'XSV+ETM', 'XSV+ETN', 'XSV+JKO'
        ]
        self.back_blank_tag = [
            '', 'IC', 'MAG', 'MAG+JX', 'MAG+XSV+EP+EC', 'MAJ', 'MM', 'MM+EC', 'NNB', 'NNB+JKB', 'NNB+VCP', 'NNB+VCP+EC', 'NNB+VCP+EF', 'NNBC', 'NNBC+VCP+EC',
            'NNG', 'NNG+JC', 'NNG+JX+JKO', 'NNG+VCP', 'NNG+VCP+EC', 'NNG+VCP+ETM', 'NNP', 'NNP+JX', 'NP', 'NP+JKG', 'NP+JKO', 'NP+JKS', 'NP+JX', 'NP+VCP+EC', 'NP+VCP+EF',
            'NR', 'SC', 'SL', 'SN', 'SSC', 'SSO', 'SY', 'VA', 'VA+EC', 'VA+EC+VX+ETM', 'VA+EF', 'VA+ETM', 'VA+ETN', 'VA+ETN+JKB+JX', 'VCN', 'VCN+EC', 'VCN+EF', 'VCN+ETM',
            'VCN+ETN', 'VCP', 'VCP+EF', 'VV', 'VV+EC', 'VV+EC+JX', 'VV+EC+VX', 'VV+EC+VX+EC', 'VV+EC+VX+EF', 'VV+EC+VX+EP+EC', 'VV+EC+VX+ETM', 'VV+EF', 'VV+EP', 'VV+EP+EC',
            'VV+EP+ETM', 'VV+ETM', 'VV+ETN', 'VV+ETN+VCP+EF', 'VX', 'VX+ETM', 'XPN', 'XR', 'XSA+ETN+VCP+EF', 'XSN'
        ]

        
    def morpheme(self, sentence_list):
        new_sentence = []
        for i, sentence in tqdm(enumerate(sentence_list)):
            temp = []
            if self.mode == 'dec':
                temp.append('sos_')
            for t in self.text_tokenizer.pos(sentence):
                temp.append('_'.join(t))
            if self.mode == 'dec':
                temp.append('eos_')
            new_sentence.append(' '.join(temp))
            
        return new_sentence

    def fit(self, sentence_list):
        for sentence in tqdm(sentence_list):
            for word in sentence.split(" "):
                try:
                    self.word_count[word] += 1
                except:
                    self.word_count[word] += 1

        self.word_count =  dict(sorted(self.word_count.items(), key = self.sort_target, reverse = True))

        self.txt2idx = {"pad":0, "unk_":1}
        self.idx2txt = {0 :"pad", 1: "unk_"}
        if self.max_vocab_size == -1 :
            for i, word in enumerate(list(self.word_count.keys())):
                self.txt2idx[word] = i+2
                self.idx2txt[i+2] =  word
        else:
            for i, word in enumerate(list(self.word_count.keys())[:self.max_vocab_size]):
                self.txt2idx[word] = i+2
                self.idx2txt[i+2] =  word
    def sort_target(self, x):
        return x[1]
    def txt2token(self, sentence_list):
        tokens = []
        for sentence in tqdm(sentence_list):
            token = [0]*self.max_length
            for i, w in enumerate(sentence.split("")):
                if i == self.max_length:
                    break
                try:
                    token[i] = self.txt2idx[w]
                except:
                    token[i] = self.txt2idx["unk_"]
            tokens.append(token)
        return np.array(tokens)
    
    def convert(self, token):
        sentence = []
        for j, i in enumerate(token):
            if self.mode == "enc":
                if i!= self.txt2idx["pad_"]:
                    sentence.append(self.idx2txt[i].split("_")[0])
            elif self.mode == "dec":
                if i != self.txt2idx["eos_"] or i != self.txt2idx["pad_"]:
                    break
            elif i != 0:
                sentence.append(self.idx2txt[i].split("_")[0])
                #앞 뒤 태그를 확인하여 띄어쓰기 추가
                if self.idx2txt.split("_")[1] in self.font_blank_tag:
                    try :
                        if self.idx2txt[token[j+1]].split("_")[1] in self.back_blank_tag:
                            sentence.append(" ")
                    except:
                        pass
        sentence = "".join(sentence)
        if self.mode == "enc":
            sentence =  sentence[:-1]
        elif self.mode == "dec":
            sentence = sentence[3:-1]
        return sentence

src_tokenizer = Mecab_Tokenizer(encoder_len, mode = "enc", max_vocab_size = max_vocab_size)
tar_tokenizer = Mecab_Tokenizer(decoder_len, mode = "dec", max_vocab_size = max_vocab_size)

train_src = src_tokenizer.morpheme(df_train.total)
val_src =  src_tokenizer.morpheme(df_val.total)
test_src = src_tokenizer(test.total)

train_tar = tar_tokenizer.morpheme(df_train.summary)
val_tar =  tar_tokenizer.morpheme(df_val.summary)

train_src_len = []
for m in train_src:
    m_len = len(m.split(' '))
    train_src_len.append(m_len)
print('train_src_max_len :', max(train_src_len))
plt.hist(train_src_len, bins=30)
plt.show()

train_tar_len = []
for m in train_tar:
    m_len = len(m.split(' '))
    train_tar_len.append(m_len)
print('train_tar_max_len :', max(train_tar_len))
plt.hist(train_tar_len, bins=30)
plt.show()

src_tokenizer.fit(train_src)
tar_tokenizer.fit(train_tar)

train_src_tokens = src_tokenizer.txt2token(train_src)
val_src_tokens = src_tokenizer.txt2idx(val_src)
test_src_tokens = src_tokenizer.txt2token(test_src)

train_tar_tokens = tar_tokenizer.txt2idx(train_tar)
val_tar_tokens =  tar_tokenizer.txt2idx(val_tar)

input_vocab_size = len(src_tokenizer.txt2idx)
target_vocab_size = len(tar_tokenizer.txt2idx)

# input_vocab_size, target_vocab_size

# df_train.summary.iloc[0]

# train_tar_tokens[0], tar_tokenizer.convert(train_tar_tokens[0])