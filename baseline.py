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
import sys
from torch.nn.modules import transformer 

# sys.path.insert(0, "/opt/homebrew/Caskroom/miniforge/base/envs/coding/lib/python3.8/site-packages")
from torch.nn.modules.activation import MultiheadAttention
from tqdm import tqdm
import glob
sys.path.insert(0, "/Users/seojiwon/Library/Python/3.8/lib/python/site-packages/")
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
decoder_len = 500
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
        # 모든 단어를 dict 형태로 저장
        for sentence in tqdm(sentence_list):
            for word in sentence.split(" "):
                try:
                    self.word_count[word] += 1
                except:
                    self.word_count[word] = 1

        self.word_count =  dict(sorted(self.word_count.items(), key = self.sort_target, reverse = True))

        self.txt2idx = {"pad_":0, "unk_":1}
        self.idx2txt = {0 :"pad_", 1: "unk_"}
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
            for i, w in enumerate(sentence.split(" ")):
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
                if i == self.txt2idx["eos_"] or i == self.txt2idx["pad_"]: #token이 eos나 pad가 아니라면
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

# self, max_length, mode, max_vocab_size=-1
src_tokenizer = Mecab_Tokenizer(encoder_len, mode = "enc", max_vocab_size = max_vocab_size) 
tar_tokenizer = Mecab_Tokenizer(decoder_len, mode = "dec", max_vocab_size = max_vocab_size)

train_src = src_tokenizer.morpheme(df_train.total) #total데이터를 순서대로 20000개(max_length)에 인덱스로 삽입
val_src =  src_tokenizer.morpheme(df_val.total) 
test_src = src_tokenizer.morpheme(test.total)

train_tar = tar_tokenizer.morpheme(df_train.summary)
val_tar =  tar_tokenizer.morpheme(df_val.summary)

train_src_len = []
for m in train_src:
    m_len = len(m.split(' '))
    train_src_len.append(m_len)
print('train_src_max_len :', max(train_src_len))
plt.hist(train_src_len, bins=30)


train_tar_len = []
for m in train_tar:
    m_len = len(m.split(' '))
    train_tar_len.append(m_len)
print('train_tar_max_len :', max(train_tar_len))
plt.hist(train_tar_len, bins=30)


src_tokenizer.fit(train_src)
tar_tokenizer.fit(train_tar)

train_src_tokens = src_tokenizer.txt2token(train_src)
val_src_tokens = src_tokenizer.txt2token(val_src)
test_src_tokens = src_tokenizer.txt2token(test_src)

train_tar_tokens = tar_tokenizer.txt2token(train_tar)
val_tar_tokens =  tar_tokenizer.txt2token(val_tar)

input_vocab_size = len(src_tokenizer.txt2idx)
target_vocab_size = len(tar_tokenizer.txt2idx)
tar_tokenizer.convert(train_tar_tokens[1])
# input_vocab_size, target_vocab_size

# df_train.summary.iloc[0]

# train_tar_tokens[0], tar_tokenizer.convert(train_tar_tokens[0])


class CustomDataset(Dataset):
    def __init__(self, src_tokens, tar_tokens, mode = "train"):
        self.mode = mode
        self.src_tokens = src_tokens
        if self.mode == "train":
            self.tar_tokens =  tar_tokens
    def __len__(self):
        return len(self.src_tokens)

    def __getitem__(self, i) :
        src_token = self.src_tokens[i]
        if self.mode == "train":
            tar_token = self.tar_tokens[i]
            return {
                "src_token" : torch.tensor(src_token, dtype = torch.long),
                "tar_token" : torch.tensor(tar_token, dtype = torch.long)
            } #딕셔너리 형태로 소스토큰과 타겟토큰을 반환
        else:
            return {"src_token" : torch.tensor(src_token, dtype =  torch.long)}



train_dataset =  CustomDataset(train_src_tokens, train_tar_tokens)
val_dataset =  CustomDataset(val_src_tokens, val_tar_tokens)
test_dataset = CustomDataset(test_src_tokens, None, "test")


train_datloader =  torch.utils.data.DataLoader(train_dataset, batch_size =  batch_size, num_workers = 1, shuffle = True)
val_dataloader =  torch.utils.data.DataLoader(val_dataset, batch_size =  batch_size, num_workers = 1, shuffle = False)
test_dataloader =  torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, num_workers = 1, shuffle =  False)


# transformer

"""get angles:
    포지셔널 인코딩을 위해 각 순서의 angle 설정"""
def get_angles(pos, i, d_model):
    angle_rates =  1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))  #d_model =  512 (임베딩 벡터 차원 수)
    return  pos * angle_rates #전체에 대해서 레이트 곱해주기


def positional_encoding(position, d_model):
    angle_rads =  get_angles(np.arange(position)[:, np.newaxis], #pos - 위치정보 리스트
                            np.arange(d_model)[np.newaxis, :], #i 차원에 대한 리스트 
                            d_model) #차원 하나 늘려주기 
    #어레이에 있는 각 홀수 인덱스마다 사인함수 걸어주기
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    #어레이에 있는 각 짝수 인덱스마다 코사인 함수 걸어주기
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return torch.tensor(pos_encoding, dtype =  torch.float32)

def create_padding_mask(seq):
    seq = torch.tensor(torch.eq(seq, 0), dtype =  torch.float32)
    # 어텐션 로짓에 패딩 차원 더해주기
    seq  = seq.unsqueeze(1).unsqueeze(2)
    return seq
    #(배치사이즈, 1,1, 시퀀스렝스)



def create_look_ahead_mask(size): #순방향 마스크 만들기
    mask =  torch.ones(size, size).triu(diagonal=1)
    return mask #(seq_len, seq_len)

def scaled_dot_product_attention(q, k, v, mask):
    #첫번째로 쿼리와 키값을 내적
    matmul_qk = torch.matmul(q, torch.transpose(k, -2, -1)) #(..., seq_len_q, seq_len_k)

    dk = k.size()[-1]
    # dk =  키 값의 임베딩 차원 수 
    scaled_attention_logits =  matmul_qk /math.sqrt(dk)
    # q와 k의 내적 (각 값들 관의 관계식)을 루트 임베딩 값으로 나눠주기(스케일링) = > 코사인 시밀러리티
    #  마스크를 붙여야 하면 = > 스케일링 된 텐서에 마스크 붙여주기
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    attention_weights = torch.nn.functional.softmax(scaled_attention_logits, dim = 1) # 어텐션 가중치를 소프트맥스로 구하기 
    output = torch.matmul(attention_weights, v)  # (..., seq_len_q, depth_v) #마지막으로 어텐션 확률과 자기 자신(v를 내적하면 끝!

    return output, attention_weights 

def print_out(q, k, v):
    temp_out, temp_attn = scaled_dot_product_attention(
      q, k, v, None)
    print('Attention weights are:')
    print(temp_attn)
    print('Output is:')
    print(temp_out)

np.set_printoptions(suppress=True)

temp_k = torch.tensor([[10, 0, 0],
                      [0, 10, 0],
                      [0, 0, 10],
                      [0, 0, 10]], dtype=torch.float32)  # (4, 3)

temp_v = torch.tensor([[1, 0],
                      [10, 0],
                      [100, 5],
                      [1000, 6]], dtype=torch.float32)  # (4, 2)

# This `query` aligns with the second `key`,
# so the second `value` is returned.
temp_q = torch.tensor([[0, 10, 0]], dtype=torch.float32)  # (1, 3)
print_out(temp_q, temp_k, temp_v)

temp_q = torch.tensor([[10, 10, 0]], dtype=torch.float32)  # (1, 3)
print_out(temp_q, temp_k, temp_v)

temp_q = torch.tensor([[0, 0, 10],
                      [0, 10, 0],
                      [10, 10, 0]], dtype=torch.float32)  # (3, 3)
print_out(temp_q, temp_k, temp_v)
sys.path.insert(0, "/Users/seojiwon/Library/Python/3.8/lib/python/site-packages")
from torch.nn import Module
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.wo = nn.Linear(d_model, d_model)

    def forward(self, v, k, q, mask):
        batch_size = q.size()[0]

        q = self.wq(q).view(batch_size, -1, self.num_heads, self.depth).transpose(1,2)
        k = self.wk(k).view(batch_size, -1, self.num_heads, self.depth).transpose(1,2)
        v = self.wv(v).view(batch_size, -1, self.num_heads, self.depth).transpose(1,2)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention =  scaled_attention.transpose(1,2).contiguous().view(batch_size, -1, self.num_heads *  self. depth)

        output = self.wo(scaled_attention) # batch size, sel_len_q, d_model

        return output, attention_weights #아웃풋 사이즈(1, 60, 512 => batch_size , depth, d_model) #어텐션 웨이트 사이즈 = (1, 8, 60, 60) (1, head, depth, 512/head)



# out.shape, attn.shape
temp_mha = MultiHeadAttention(d_model = 512, num_heads = 8)
y =  torch.rand(1, 60, 512)
out, attn =  temp_mha.forward(y, k = y,  q = y, mask =  None)

class FFN(nn.Module):
    def __init__(self, d_model, dff):
        super(FFN, self).__init__()
        self.layer1 = nn.Linear(d_model, dff)
        self.activation =  nn.ReLU()
        self.fc = nn.Linear(dff, d_model)


    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.fc(x)

        return x
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, maximum_position_encoding, rate = 0.1):
        super(EncoderLayer, self).__init__()
        
        self.mha  =  MultiheadAttention(d_model, num_heads) #512, 8
        self.ffn = FFN(d_model, dff) #512, 2048

        self.layernorm1 = nn.LayerNorm([maximum_position_encoding, d_model]) #500, 512
        self.layernorm2 = nn.LayerNorm([maximum_position_encoding, d_model]) #500, 512

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x, mask):
        attn_output, _ = self.mha(x,x,x, mask)
        attn_output =  self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output) #residual connection

        ffn_output =  self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output) #concat

        return out2

sample_encoder_layer = EncoderLayer(512, 8, 2048, encoder_len) #d_model, num_heads, dff, maximum_position_encoding
sample_encoder_layer_output =  sample_encoder_layer(
    torch.rand(64, encoder_len, 512), None
)



class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, maximum_position_encoding, rate = 0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiheadAttention(d_model, num_heads)
        self.mha2 = MultiheadAttention(d_model, num_heads)

        self.ffn = FFN(d_model, dff)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
        self.dropout3 = nn.Dropout(rate)

        self.layernorm1 = nn.ModuleList([copy.deepcopy(nn.LayerNorm([i+1, d_model])) for i in range(maximum_position_encoding)])
        self.layernorm2 = nn.ModuleList([copy.deepcopy(nn.LayerNorm([i+1, d_model])) for i in range(maximum_position_encoding)])
        self.layernorm3 = nn.ModuleList([copy.deepcopy(nn.LayerNorm([i+1, d_model])) for i in range(maximum_position_encoding)])



#     def forward(self, x, enc_output, look_ahead_mask, padding_mask):
#         # enc_output.shape == (batch_size, input_seq_len, d_model)
#         attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
#         attn1 = self.dropout1(attn1)
#         out1 = self.layernorms1[x.size(1)-1](attn1 + x)
        
#         attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
#         attn2 = self.dropout2(attn2)
#         out2 = self.layernorms2[x.size(1)-1](attn2 + out1)  # (batch_size, target_seq_len, d_model)
        
#         ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
#         ffn_output = self.dropout3(ffn_output)
#         out3 = self.layernorms3[x.size(1)-1](ffn_output + out2)  # (batch_size, target_seq_len, d_model)
        
#         return out3, attn_weights_block1, attn_weights_block2
    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x,x,x,look_ahead_mask)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1[x.size(1)-1](attn1 + x)

        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2[x.size(1)-1](attn2+out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3[x.size(1)-1](ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2


        

sample_decoder_layer = DecoderLayer(512, 8, 2048, decoder_len)
sample_decoder_layer_output, _, _ = sample_decoder_layer(
    torch.rand(64, decoder_len, 512), sample_encoder_layer_output, 
    None, None
)

# def clones(module, N):
#     return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module): #여러개의 인코더 레이어를 쌓기 위한 클래스
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, device, rate = 0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        #  self.pos_encoding = positional_encoding(maximum_position_encoding, d_model).to(device)
        self.dec_layers = clones(EncoderLayer(d_model, num_heads, dff, maximum_position_encoding, rate), num_layers)
        self.dropout = nn.Dropout(rate)
        
    def forward(self, x, mask, enc_output=None):
        if enc_output == None:
            seq_len = x.size()[1]
            attention_weights = {}
            x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
            x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
            x += self.pos_encoding[:, :seq_len, :]
            x = self.dropout(x)
            for i in range(self.num_layers):
                x = self.dec_layers[i](x, mask) #클론된 인코더 레이어임 
        else:
            x = enc_output
            
        return x

sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8,
                         dff=2048, input_vocab_size=input_vocab_size,
                         maximum_position_encoding=encoder_len,
                         device='cpu')

temp_input = torch.randint(low=0, high=input_vocab_size, size=(64, encoder_len))

sample_encoder_output = sample_encoder(temp_input, mask=None, enc_output=None)

print(sample_encoder_output.shape)  # (batch_size, input_seq_len, d_model)

# class Decoder(nn.Module):
#     def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, device, rate=0.1):
#         super(Decoder, self).__init__()

#         self.d_model = d_model
#         self.num_layers = num_layers

#         self.embedding = nn.Embedding(target_vocab_size, d_model)
#         self.pos_encoding = positional_encoding(maximum_position_encoding, d_model).to(device)
        
#         self.dec_layers = clones(DecoderLayer(d_model, num_heads, dff, maximum_position_encoding, rate), num_layers)
#         self.dropout = nn.Dropout(rate)

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_posiotion_encoding, device, rate = 0.1):
        super(Decoder, self).__init__()

        self.d_model =  d_model
        self.num_layers =  num_layers

        self.embedding = nn.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_posiotion_encoding, d_model)

        self.dec_layers = clones(DecoderLayer(d_model, num_heads, dff, maximum_posiotion_encoding, rate), num_layers)
        self.dropout = nn.Dropout(rate)


#     def forward(self, x, enc_output, look_ahead_mask, padding_mask):
#         seq_len = x.size()[1]
#         attention_weights = {}
#         x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
#         x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
#         x += self.pos_encoding[:, :seq_len, :]
#         x = self.dropout(x)
        
#         for i in range(self.num_layers):
#             x, block1, block2 = self.dec_layers[i](x, enc_output, look_ahead_mask, padding_mask)

#             attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
#             attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
            
#         # x.shape == (batch_size, target_seq_len, d_model)
#         return x, attention_weights
    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        seq_len = x.size()[1]
        attention_weights = {} 
        x = self.embedding(x)
        x *= torch.sqrt(torch.tensor(self.d_model, dtype = torch.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, look_ahead_mask, padding_mask)

            attention_weights["decoder_layer{}_block1".format(i+1)] = block1
            attention_weights["decoder_layer{}_block2".format(i+1)] =  block2
        return x , attention_weights

# sample_decoder = Decoder(num_layers=2, d_model=512, num_heads=8,
#                          dff=2048, target_vocab_size=target_vocab_size,
#                          maximum_position_encoding=decoder_len,
#                          device='cpu')
sample_decoder = Decoder(num_layers = 2, d_model = 512, num_heads = 8,
                        dff = 2048, target_vocab_size = target_vocab_size,
                        maximum_posiotion_encoding=decoder_len,
                        device = "cpu")
# temp_input = torch.randint(low=0, high=target_vocab_size, size=(64, decoder_len))
temp_input = torch.randint(low=0, high = target_vocab_size, size = (64, decoder_len))

# output, attn = sample_decoder(temp_input,
#                               enc_output=sample_encoder_output,
#                               look_ahead_mask=None,
#                               padding_mask=None)

output, attn = sample_decoder(temp_input,
                              enc_output =  sample_encoder_output,
                              look_ahead_mask = None,
                              padding_mask =  None)

output.shape, attn['decoder_layer2_block2'].shape

# class Transformer(nn.Module):
#     def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
#                target_vocab_size, pe_input, pe_target, device, rate=0.1):
#         super().__init__()
#         self.device = device
#         self.encoder = Encoder(num_layers, d_model, num_heads, dff,
#                                  input_vocab_size, pe_input, device, rate)

#         self.decoder = Decoder(num_layers, d_model, num_heads, dff,
#                                target_vocab_size, pe_target, device, rate)

#         self.final_layer = nn.Linear(d_model, target_vocab_size)

class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                target_vocab_size, pe_input, pe_target, device, rate = 0.1):
        super().__init__()
        self.device =  device
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                        input_vocab_size, pe_target, device, rate)
        self.decoder =  Decoder(num_layers, d_model, num_heads, dff, 
                                target_vocab_size, pe_target, device, rate)
        self.final_layer =  nn.Linear(d_model, target_vocab_size)


#     def forward(self, inputs):
#         inp, tar, enc_output = inputs

#         enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(inp, tar)

#         enc_output = self.encoder(inp, enc_padding_mask, enc_output)  # (batch_size, inp_seq_len, d_model)

#         # dec_output.shape == (batch_size, tar_seq_len, d_model)
#         dec_output, attention_weights = self.decoder(
#             tar, enc_output, look_ahead_mask, dec_padding_mask)

#         final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

#         return final_output, attention_weights, enc_output

    def forward(self, inputs):
        inp, tar, enc_output =  inputs

        enc_padding_mask, look_ahead_mask, dec_padding_mask =  self.create_masks(inp, tar)

        enc_output = self.encoder(inp, enc_padding_mask, enc_output)

        dec_output, attention_weights = self.decoder(tar, enc_output, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)

        return final_output, attention_weights, enc_output

#     def create_masks(self, inp, tar):
#         # Encoder padding mask
#         enc_padding_mask = create_padding_mask(inp)

#         # Used in the 2nd attention block in the decoder.
#         # This padding mask is used to mask the encoder outputs.
#         dec_padding_mask = create_padding_mask(inp)

#         # Used in the 1st attention block in the decoder.
#         # It is used to pad and mask future tokens in the input received by
#         # the decoder.
#         look_ahead_mask = create_look_ahead_mask(tar.size(1))
#         dec_target_padding_mask = create_padding_mask(tar)
#         look_ahead_mask = torch.maximum(dec_target_padding_mask.to(self.device), look_ahead_mask.to(self.device))

#         return enc_padding_mask, look_ahead_mask, dec_padding_mask

    def create_masks(self, inp, tar):
        enc_padding_mask = create_padding_mask(inp)

        dec_padding_mask = create_padding_mask(inp)

        look_ahead_mask = create_look_ahead_mask(tar.size(1))
        dec_target_padding_mask =  create_padding_mask(tar)
        look_ahead_mask = torch.maximum(dec_target_padding_mask.to(self.device), look_ahead_mask.to(self.device))

        return enc_padding_mask, look_ahead_mask, dec_padding_mask

# transformer = Transformer(
#     num_layers=num_layers,
#     d_model=d_model,
#     num_heads=num_heads,
#     dff=dff,
#     input_vocab_size=input_vocab_size,
#     target_vocab_size=target_vocab_size,
#     pe_input=encoder_len,
#     pe_target=decoder_len-1,
#     device=device,
#     rate=dropout_rate
# )
transformer = Transformer(
    num_layers = num_layers, 
    d_model =  d_model,
    num_heads= num_heads,
    dff = dff,
    input_vocab_size = input_vocab_size, 
    target_vocab_size = target_vocab_size,
    pe_input =  encoder_len,
    pe_target= decoder_len -1, 
    device = device,
    rate = dropout_rate
)
# transformer = transformer.to(device)
# optimizer = torch.optim.Adam(transformer.parameters(), lr=learning_rate)
# criterion = nn.CrossEntropyLoss()
## 손실함수 및 평가함수 정의

optimizer =  torch.optim.Adam(transformer.parameters(), lr  = learning_rate)
criterion = nn.CrossEntropyLoss()


# def loss_function(real, pred):
#     mask = torch.logical_not(torch.eq(real, 0))
#     loss_ = criterion(pred.permute(0,2,1), real)
#     mask = torch.tensor(mask, dtype=loss_.dtype)
#     loss_ = mask * loss_

#     return torch.sum(loss_)/torch.sum(mask) 
def loss_function(real, pred):
    mask = torch.logical_not(torch.eq(real, 0))
    loss_ = criterion(pred.permute(0,2,1), real)
    mask = torch.tensor(mask, dtype = loss_.dtype)
    loss_ = mask * loss_

    return torch.sum(loss_)/torch.sum(mask)

# def accuracy_function(real, pred):
#     accuracies = torch.eq(real, torch.argmax(pred, dim=2))
#     mask = torch.logical_not(torch.eq(real, 0))
#     accuracies = torch.logical_and(mask, accuracies)
#     accuracies = torch.tensor(accuracies, dtype=torch.float32)
#     mask = torch.tensor(mask, dtype=torch.float32)
    
#     return torch.sum(accuracies)/torch.sum(mask)

def accuracy_function(real, pred):
    accuracies = torch.eq(real, torch.argmax(pred, dim =2))
    mask = torch.logical_not(torch.eq(real, 0))
    accuracies = torch.logical_and(mask, accuracies)
    accuracies = torch.tensor(accuracies, dtype = torch.float32)

    return torch.sum(accuracies)/ torch.sum(mask)
## 학습 정의


# def train_step(batch_item, epoch, batch, training):
#     src = batch_item['src_token'].to(device)
#     tar = batch_item['tar_token'].to(device)
    
#     tar_inp = tar[:, :-1]
#     tar_real = tar[:, 1:]
    
#     if training is True:
#         transformer.train()
#         optimizer.zero_grad()
#         with torch.cuda.amp.autocast():
#             output, _, _ = transformer([src, tar_inp, None])
#             loss = loss_function(tar_real, output)
#         acc = accuracy_function(tar_real, output)
#         loss.backward()
#         optimizer.step()
#         lr = optimizer.param_groups[0]["lr"]
#         return loss, acc, round(lr, 10)
def train_step(batch_item, epoch, batch, training):
    src = batch_item["src_token"].to(device)
    tar = batch_item["tar_token"].to(device)

    tar_inp =  tar[:, :-1]
    tar_real =  tar[:, 1:]

    if training is True:
        transformer.train()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output, _, _ = transformer([src, tar_inp, None])
            loss = loss_function(tar_real, output)
        acc = accuracy_function(tar_real, output)
        loss.backward()
        optimizer.step()
        lr = optimizer.param_groups[0]["lr"]
        return loss, acc, round(lr, 10)

#     else:
#         transformer.eval()
#         with torch.no_grad():
#             output, _, _ = transformer([src, tar_inp, None])
#             loss = loss_function(tar_real, output)
#         acc = accuracy_function(tar_real, output)
#         return loss, acc
    else:
        transformer.eval()
        with torch.no_grad():
            output, _, _ = transformer([src, tar_inp, None])
            loss = loss_function(tar_real, output)
        acc = accuracy_function(tar_real, output)
        return loss, acc
# transformer.to(device)
## 학습

# loss_plot, val_loss_plot = [], []
# acc_plot, val_acc_plot = [], []

loss_plot, val_loss_plot = [], []
acc_plot, val_acc_plot = [], []
# for epoch in range(epochs):
#     gc.collect()
#     total_loss, total_val_loss = 0, 0
#     total_acc, total_val_acc = 0, 0
for epoch in range(epochs):
    gc.collect()
    total_loss, total_val_loss = 0, 0
    total_acc, total_val_acc = 0, 0    
#     tqdm_dataset = tqdm(enumerate(train_dataloader))
#     training = True
#     for batch, batch_item in tqdm_dataset:
#         batch_loss, batch_acc, lr = train_step(batch_item, epoch, batch, training)
#         total_loss += batch_loss
#         total_acc += batch_acc
    tqdm_dataset = tqdm(enumerate(train_datloader))
    training = True
    for batch, batch_item in tqdm_dataset:
        batch_loss, batch_acc,  lr = train_step(batch_item, epoch, batch, training)
        total_loss += batch_loss
        total_acc += batch_acc
        tqdm_dataset.set_postfix({
            'Epoch': epoch + 1,
            'LR' : lr,
            'Loss': '{:06f}'.format(batch_loss.item()),
            'Total Loss' : '{:06f}'.format(total_loss/(batch+1)),
            'Total ACC' : '{:06f}'.format(total_acc/(batch+1))
        })
        # tqdm_dataset.set_postfix({
        #     "Epoch" :  epoch + 1, 
        #     "LR" : lr, 
        #     "Loss" : '{:06f}'.format(batch_loss.item()),
        #     "Total Loss" : '{:06f}'.format(total_loss/batch + 1)),
        #     'Total ACC' : '{:06f}'.format(total_acc/(batch+1))
        # })
    loss_plot.append(total_loss/(batch+1))
    acc_plot.append(total_acc/(batch+1))
    
#     tqdm_dataset = tqdm(enumerate(val_dataloader))
#     training = False
#     for batch, batch_item in tqdm_dataset:
#         batch_loss, batch_acc = train_step(batch_item, epoch, batch, training)
#         total_val_loss += batch_loss
#         total_val_acc += batch_acc
        
#         tqdm_dataset.set_postfix({
#             'Epoch': epoch + 1,
#             'Val Loss': '{:06f}'.format(batch_loss.item()),
#             'Total Val Loss' : '{:06f}'.format(total_val_loss/(batch+1)),
#             'Total Val ACC' : '{:06f}'.format(total_val_acc/(batch+1))
#         })
#     val_loss_plot.append(total_val_loss/(batch+1))
#     val_acc_plot.append(total_val_acc/(batch+1))
    tqdm_dataset =  tqdm(enumerate(val_dataloader))
    training = False
    for batch, batch_item in tqdm_dataset:
        batch_loss, batch_acc = train_step(batch_item, epoch, batch, training)
        total_val_loss += batch_loss
        total_val_acc += batch_acc
        
        tqdm_dataset.set_postfix({
            'Epoch': epoch + 1,
            'Val Loss': '{:06f}'.format(batch_loss.item()),
            'Total Val Loss' : '{:06f}'.format(total_val_loss/(batch+1)),
            'Total Val ACC' : '{:06f}'.format(total_val_acc/(batch+1))
        })

        val_loss_plot.append(total_val_loss/(batch+1))
        val_acc_plot.append(total_val_acc/(batch+1))