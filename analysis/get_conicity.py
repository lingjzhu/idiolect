#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import seaborn as sns
import pandas as pd
import torch
import transformers
import numpy as np
import gensim
from collections import defaultdict

from torch import nn
import torch.nn.functional as F
from transformers import RobertaConfig, BertTokenizerFast, BertModel, RobertaTokenizer, RobertaModel, AdamW,RobertaTokenizerFast
from tqdm import tqdm

from model_amazon_self_attention import SRoberta,DNNSelfAttention,AttentionPooling



def get_conicity(vectors):
    mean_vec = torch.mean(vectors,dim=0)
    mean_vec = F.normalize(mean_vec,dim=-1).unsqueeze(-1)
    atm = torch.matmul(vectors,mean_vec)
    conicity = torch.mean(atm)
    return atm,conicity


def extract_emb(text):
    tokenized = tokenizer.encode_plus(text,add_special_tokens=True, max_length=102,truncation=True,return_tensors="pt")
    hidden = model(tokenized['input_ids'].to(device),tokenized['attention_mask'].to(device))
    hidden = F.normalize(hidden,dim=-1)
#    hidden = hidden.cpu().detach().numpy()
    return hidden
	
	
	
device = 'cuda'


model_path = '/gpfs/accounts/lingjzhu_root/lingjzhu1/lingjzhu/authorship_models/redditroberta-cosine-modified_anchor-mask-0.1-delta-0.4-0.6-alpha-30.0/model-4'

model = torch.load(model_path)
model.eval()
model.to(device)

tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')



data = pd.read_csv('/gpfs/accounts/lingjzhu_root/lingjzhu1/lingjzhu/authorship/reddit_test_samples',sep='\t')
users = data['user'].unique()


results = defaultdict(list)

with torch.no_grad():
    for user in tqdm(users):

        subset = data[data['user']==user]
        texts = subset['text'].tolist()

        embeddings = [extract_emb(text) for text in texts]
        embeddings = torch.cat(embeddings,dim=0)

        atm, conicity = get_conicity(embeddings)

        results['user'].append(user)
        results['conicity'].append(float(conicity.cpu().detach().numpy()))
		
		
out = pd.DataFrame.from_dict(results)
out.to_csv('/gpfs/accounts/lingjzhu_root/lingjzhu1/lingjzhu/authorship/reddit_conicity',index=None)






