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



def compute_distinctiveness(vectors):

    distinct = []
    for vec in tqdm(vectors):
        cosines = torch.matmul(vec,vectors.transpose(1,0))
        cosines = cosines.cpu().detach().numpy()
        neighbors = np.sum(np.where(cosines>0.5,1,0))
        distinct.append(neighbors)
        
    distinct = np.array(distinct)
    distinct = 1-distinct/len(distinct)
    return distinct

def extract_emb(text):
    tokenized = tokenizer.encode_plus(text,add_special_tokens=True, max_length=102,truncation=True,return_tensors="pt")
    hidden = model(tokenized['input_ids'].to(device),tokenized['attention_mask'].to(device))
    hidden = F.normalize(hidden,dim=-1)
    hidden = hidden.cpu().detach().numpy()
    return hidden
    
    

device = 'cuda'


#model_path = '/gpfs/accounts/lingjzhu_root/lingjzhu1/lingjzhu/authorship_models/redditroberta-cosine-modified_anchor-mask-0.1-delta-0.4-0.6-alpha-30.0/model-4'

model_path = '/gpfs/accounts/lingjzhu_root/lingjzhu1/lingjzhu/authorship_models/final-roberta-cosine-modified_anchor-mask-0.1-delta-0.4-0.6-alpha-30.0/model-5'

model = torch.load(model_path)
model.eval()
model.to(device)

tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

    


data = pd.read_csv('/gpfs/accounts/lingjzhu_root/lingjzhu1/lingjzhu/authorship/amazon_test_samples',sep='\t')

with torch.no_grad():
    for round in range(5):
        samples = data.groupby('user').sample(1)

        users = samples['user']
        texts = samples['text']


        embeddings = np.zeros((len(users),768))

        for i,(u,t) in tqdm(enumerate(zip(users,texts))):
            embeddings[i,:] = extract_emb(t)
            
            
            
        distinct = compute_distinctiveness(torch.tensor(embeddings).to(device))    
            
        output = pd.DataFrame()
        output['user'] = users
        output['text'] = texts
        output['distinct'] = distinct   
        output.to_csv('/gpfs/accounts/lingjzhu_root/lingjzhu1/lingjzhu/authorship/amazon_distinct_%s'%(round),sep='\t',index=None)
            
    
    