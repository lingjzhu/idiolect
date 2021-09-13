#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 17:06:21 2020

"""


import pandas as pd
import torch
import transformers
import numpy as np
import argparse
import os

from torch import nn
from transformers.modeling_bert import BertLayerNorm, gelu
from transformers.modeling_roberta import RobertaPreTrainedModel
from transformers import RobertaConfig, RobertaTokenizer, RobertaModel, AdamW,RobertaTokenizerFast
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score
from torch.nn import BCEWithLogitsLoss




# preprocess the texts
def loads(path):
    data = pd.read_csv(path,sep="\t\t",header=None)
    
    labels = data[4].tolist()
    texts = [(i,j) for i,j in zip(data[2].tolist(),data[3].tolist())]
    users = data[0].tolist()
    products = data[1].tolist()
    return texts, labels, users, products

#text_tr, text_te, label_tr, label_te = train_test_split(texts,labels,test_size=0.1)

def preprocess(texts):
    
    pair_a = []
    pair_b = []
    for i in texts:
        a, b = i
        pair_a.append(a)
        pair_b.append(b)
    
    return pair_a, pair_b



def tokenize(textA, textB,seq_len=100):
    
    length = len(textA)
    input_ids = torch.ones(length,seq_len*2).long()
    attention_mask = torch.zeros(length,seq_len*2).long()
    
    for k, a, b in tqdm(zip(range(length),textA, textB)):
        seqA = tokenizer.encode_plus(a,add_special_tokens=True, max_length=seq_len,truncation=True,return_tensors="pt")
        seqB = tokenizer.encode_plus(b,add_special_tokens=True, max_length=seq_len,truncation=True,return_tensors="pt")
        lengthA = seqA['input_ids'].shape[1]
        lengthB = seqB['input_ids'].shape[1]
        if args.permute_words==True:
            seqA['input_ids'] = seqA['input_ids'][:,torch.randperm(seqA['input_ids'].size(1))]
            seqB['input_ids'] = seqB['input_ids'][:,torch.randperm(seqB['input_ids'].size(1))]

#        seqB['input_ids'][0] = 2
        input_ids[k,:lengthA] = seqA['input_ids']
        input_ids[k,lengthA:lengthA+lengthB] = seqB['input_ids']
        input_ids[k,lengthA] = 2
        attention_mask[k,:lengthA] = seqA['attention_mask']
        attention_mask[k,lengthA:lengthB+lengthA] = seqB['attention_mask']
    return input_ids,attention_mask


def batchify(text_a,attn_a,labels,size=16,shuffling=False):
    length = len(text_a)
    
    if shuffling == True:
        text_a,attn_a,labels = shuffle(text_a,attn_a,labels)
    for i in range(0,length,size):
        sampleA = text_a[i:min(i+size,length)]
#        sampleB = text_b[last:i]
        sample_attn_a = attn_a[i:min(i+size,length)]
#        sample_attn_b = attn_b[last:i]
        y = labels[i:min(i+size,length)]

        yield sampleA, sample_attn_a, torch.tensor(y).long()


#pair,attn, y = next(iter(batchify(train_a, tr_attn_a, label_tr)))



#initiate the models

#model = RobertaModel.from_pretrained("roberta-base",return_dict=True).to(device)

def masking(text,mlm_prob=0.1,mask=50264):
    
    indices_replaced = torch.bernoulli(torch.full(text.shape, mlm_prob)).bool()
    text[indices_replaced] = mask
    return text
    

class LinearHead(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base",return_dict=True)
        self.linear = nn.Linear(768,1)
        
    def forward(self,texts, masks):
        
        hidden = self.roberta(input_ids=texts,attention_mask=masks)
        hidden = hidden.last_hidden_state[:,0,:]
        logit = self.linear(hidden).squeeze(1)
        return logit




def train(args,model,train_a, tr_attn_a,label_tr):

    model.train()
    for i, (pair, attn, y) in tqdm(enumerate(batchify(train_a,tr_attn_a, label_tr,size=args.training_bsz))):
        if not args.mask_prob==0:
            pair = masking(pair,mlm_prob=args.mask_prob)
        logit = model(pair.to(device),attn.to(device))
        loss = loss_fn(logit,y.to(device).float())
        loss.backward()
        if i%args.grad_acc==0:
            optimizer.step()
            optimizer.zero_grad()
            print("The loss is: %s"%loss.item())


#evaluate
def evaluate(args,test_a, te_attn_a, label_te):
    logits = []
    targets = []
    model.eval()
    with torch.no_grad():
        for i, (pair, attn, y) in tqdm(enumerate(batchify(test_a,te_attn_a, label_te,size=args.test_bsz))):
              
                logit = model(pair.to(device),attn.to(device))
                
                logit = torch.sigmoid(logit)
                logits += list(logit.detach().cpu().numpy())
                targets += list(y.detach().cpu().numpy())
        
    return logits, targets


def compute_metric(logits, targets, threshold=0.5):
    scores = [1 if i>threshold else 0 for i in logits]        
    accuracy = (np.array(targets)==np.array(scores))
    accuracy = sum(accuracy)/len(accuracy)
    f1 = f1_score(targets,scores)
    auc = roc_auc_score(targets, logits)
    return accuracy, f1, auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",default='roberta-base',type=str)
    parser.add_argument("--training_data",default=None,type=str)
    parser.add_argument("--develop_data",default=None,type=str)
    parser.add_argument("--test_data",default=None,type=str)
    parser.add_argument("--train",action='store_true')
    parser.add_argument("--test",action='store_true')
    parser.add_argument("--save_model",default='./model',type=str)
    parser.add_argument("--load_model",default=None,type=str)
    parser.add_argument("--save_cache",default=None,type=str)
    parser.add_argument("--load_cache",default=None,type=str)
    parser.add_argument("--epochs",default=3,type=int)
    parser.add_argument("--grad_acc",default=5,type=int)
    parser.add_argument("--mask_prob",default=0.1,type=float)
    parser.add_argument("--lr",default=1e-5,type=float)
    parser.add_argument('--training_bsz',default=48,type=int)
    parser.add_argument('--test_bsz',default=90,type=int)
    parser.add_argument('--save_test_data',action='store_true')
    parser.add_argument('--permute_words',action='store_true')
    args = parser.parse_args()
    
    device = "cuda"
    
    
    #initiate the tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        

    if args.load_model:
        model = torch.load(args.load_model).to(device)
    else:
        model = LinearHead().to(device)
    
    
    optimizer = AdamW(model.parameters(),lr=args.lr)
    loss_fn = BCEWithLogitsLoss()
    
    
    model_name = "sper-reddit-model-concat-mask-%s"%(args.mask_prob)
    if args.permute_words:
        model_name = 'wpermute'+model_name
    if not os.path.exists(os.path.join(args.save_model,model_name)):
        os.mkdir(os.path.join(args.save_model,model_name))

    if args.train:
        if args.save_cache:
        
            text_tr, label_tr,_,_ = loads(args.training_data)
            text_te, label_te,_,_ = loads(args.develop_data)
            
            
            trainA, trainB = preprocess(text_tr)
            testA, testB = preprocess(text_te)
        
                
            train_a,tr_attn_a = tokenize(trainA, trainB)
            test_a,te_attn_a = tokenize(testA, testB)
            
            torch.save((train_a,tr_attn_a,label_tr),os.path.join(args.save_cache,'train'))
            torch.save((test_a,te_attn_a,label_te),os.path.join(args.save_cache,'dev'))
            
        elif args.load_cache:
            train_a,tr_attn_a,label_tr = torch.load(os.path.join(args.load_cache,'train'))
            test_a,te_attn_a,label_te = torch.load(os.path.join(args.load_cache,'dev'))
 
 
        
        for k in range(args.epochs):
            with open(os.path.join(args.save_model,model_name,'results'),'a+') as out:
                train(args,model,train_a,tr_attn_a,label_tr)
                logits,targets = evaluate(args,test_a,te_attn_a,label_te)
                accuracy,f1,auc = compute_metric(logits,targets,threshold=0.5)
                torch.save(model,os.path.join(args.save_model,model_name,'model-%s'%k))
                out.write("Epoach:%s-Acc:%s-F1:%s-AUC:%s\n"%(k,round(accuracy,3),round(f1,3),round(auc,3)))
   

    if args.test:
        # load data
        text_te, label_te, users, products = loads(args.test_data)
        if args.save_cache:
            
    #        text_te, label_te = shuffle(text_te, label_te)
            testA, testB = preprocess(text_te)
            test_a,te_attn_a = tokenize(testA, testB)
            torch.save((test_a,te_attn_a),os.path.join(args.save_cache,'test'))
            
        elif args.load_cache:
            test_a,te_attn_a = torch.load(os.path.join(args.save_cache,'test'))

        
        logits,targets = evaluate(args,test_a,te_attn_a,label_te)
        accuracy,f1,auc = compute_metric(logits,targets,threshold=0.5)

        with open(os.path.join(args.save_model,model_name,'results'),'a+') as out:
            out.write('Test-%s\n'%(args.test_data))
            out.write("Acc:%s-F1:%s-AUC:%s\n"%(round(accuracy,3),round(f1,3),round(auc,3)))
        if args.save_test_data:
            with open(os.path.join(args.save_model,model_name,'full_evaluation'),'w') as out:
                for u, p, logit, target in zip(users,products,logits,targets):
                    out.write("%s\t\t%s\t\t%s\t\t%s\n"%(u,p,logit,target))   
    


















