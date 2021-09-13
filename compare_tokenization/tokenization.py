#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 12:20:43 2020

"""

import os
import pandas as pd
import torch
import transformers
import numpy as np
import argparse
import re

from torch import nn
import torch.nn.functional as F
from transformers import AdamW,RobertaTokenizerFast, BertTokenizerFast
from nltk import word_tokenize
from sklearn.utils import shuffle
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from torch.nn.utils import clip_grad_norm_
from torch.optim import lr_scheduler

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



def tokenize(textA,textB,seq_len=100):
    
    length = len(textA)
    if re.search(r'roberta',args.tokenization):
        input_ids_a = torch.ones(length,seq_len).long()
        input_ids_b = torch.ones(length,seq_len).long()
    elif re.search(r'bert',args.tokenization):
        input_ids_a = torch.zeros(length,seq_len).long()
        input_ids_b = torch.zeros(length,seq_len).long()        
    lengths_A = np.zeros(length)
    lengths_B = np.zeros(length)
    
    for k, a, b in tqdm(zip(range(length),textA, textB)):
        seqA = tokenizer.encode_plus(a,add_special_tokens=False, max_length=seq_len,truncation=True,return_tensors="pt")
        seqB = tokenizer.encode_plus(b,add_special_tokens=False, max_length=seq_len,truncation=True,return_tensors="pt")
        input_ids_a[k,:seqA['input_ids'].shape[1]] = seqA['input_ids']
        input_ids_b[k,:seqB['input_ids'].shape[1]] = seqB['input_ids']
        lengths_A[k] = seqA['input_ids'].shape[1]
        lengths_B[k] = seqB['input_ids'].shape[1]
    return input_ids_a,input_ids_b, lengths_A, lengths_B



def tokenize_word(textA,textB,seq_len=100):
    
    length = len(textA)
    input_ids_a = torch.zeros(length,seq_len).long()
    input_ids_b = torch.zeros(length,seq_len).long()        
    lengths_A = np.zeros(length)
    lengths_B = np.zeros(length)
    
    for k, a, b in tqdm(zip(range(length),textA, textB)):
        seqA = [word_dict.get(i,1) for i in word_tokenize(' '.join(a.split(' ')[:100]))][:100]
        seqB = [word_dict.get(i,1) for i in word_tokenize(' '.join(b.split(' ')[:100]))][:100]
        input_ids_a[k,:len(seqA)] = torch.tensor(seqA)
        input_ids_b[k,:len(seqB)] = torch.tensor(seqB)
        lengths_A[k] = len(seqA)
        lengths_B[k] = len(seqB)
    return input_ids_a,input_ids_b, lengths_A, lengths_B




def batchify(text_a,text_b,lengths_a,lengths_b,labels,size=16,shuffling=False):
    length = len(text_a)
    
    if shuffling == True:
        text_a,text_b,labels = shuffle(text_a,text_b,labels)
        
    for i in range(0,length,size):
        sampleA = text_a[i:min(i+size,length)]
        sampleB = text_b[i:min(i+size,length)]
        lena = lengths_a[i:min(i+size,length)]
        lena = lengths_b[i:min(i+size,length)]
        y = labels[i:min(i+size,length)]

        yield sampleA, sampleB,torch.tensor(lena),torch.tensor(lena), torch.tensor(y).long()





def contrastive_loss(diff, target, tau_low=0.2, tau_high=0.8):
    
    within = target*torch.max(diff-tau_low,torch.tensor(0.0).to(device))**2
    without = (1-target)*torch.max(tau_high-diff,torch.tensor(0.0).to(device))**2
    
    return torch.mean(within)+torch.mean(without)



class RNN(nn.Module):
    
    def __init__(self,hidden_dim,vocab,out_dim):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab,hidden_dim)
        self.lstm = nn.LSTM(hidden_dim,hidden_dim,bidirectional=True,num_layers=2,batch_first=True)
        self.linear = nn.Sequential(nn.Linear(2*hidden_dim,hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim,out_dim))
        
        
    def forward(self, texts, lens):
        
        out = self.embedding(texts)
        packed_input = pack_padded_sequence(out, lens, batch_first=True,enforce_sorted=False)
        packed_output, (ht, ct)= self.lstm(packed_input)
#        out, _ = pad_packed_sequence(packed_output, batch_first=True)
        out = torch.cat((ht[-2,:,:], ht[-1,:,:]), dim = 1)
        out = self.linear(out)
        return out

    def predict(self,texts):
        out = self.embedding(texts)
        packed_output, (ht, ct)= self.lstm(out)
#        out, _ = pad_packed_sequence(packed_output, batch_first=True)
        out = torch.cat((ht[-2,:,:], ht[-1,:,:]), dim = 1)
        out = self.linear(out)
        return out



def train(args,model,train_a,train_b,tr_attn_a, tr_attn_b, label_tr):

    model.train()
    for i, (pair_a, pair_b, lena, lenb, y) in tqdm(enumerate(batchify(train_a,train_b,tr_attn_a, tr_attn_b, label_tr,size=args.training_bsz,shuffling=True))):
        
       
        hiddenA = model(pair_a.to(device),lena)
        hiddenB = model(pair_b.to(device),lenb)

        
        if args.distance == 'cosine':
            hiddenA = F.normalize(hiddenA,dim=1)
            hiddenB = F.normalize(hiddenB,dim=1)
            logit = 1.0-torch.sum(hiddenA*hiddenB,dim=1)
        elif args.distance == 'euclidean':
            logit = torch.norm(hiddenA - hiddenB,dim=1)
            
        
        loss = loss_fn(logit,y.to(device).float(),tau_low=args.tau_low,tau_high=args.tau_high)
        
        loss.backward()
        clip_grad_norm_(model.parameters(),1)
        if i%args.grad_acc==0:
            
            optimizer.step()
            optimizer.zero_grad()
            print("The loss is: %s"%loss.item())
        
        
                

def evaluate(args,test_a,test_b,te_attn_a, te_attn_b, label_te):
    #evaluate
    logits = []
    targets = []
    model.eval()
    for i, (pair_a, pair_b, lena, lenb, y) in tqdm(enumerate(batchify(test_a,test_b,te_attn_a, te_attn_b, label_te,size=args.test_bsz))):

        with torch.no_grad():
            hiddenA = model(pair_a.to(device),lena)
            hiddenB = model(pair_b.to(device),lenb)
            
            
            if args.distance == 'cosine':
                hiddenA = F.normalize(hiddenA,dim=1)
                hiddenB = F.normalize(hiddenB,dim=1)
                logit = 1.0-torch.sum(hiddenA*hiddenB,dim=1)
            elif args.distance == 'euclidean':
                logit = torch.norm(hiddenA - hiddenB,dim=1)
                
            logits += list(logit.detach().cpu().numpy())
            targets += list(y.detach().cpu().numpy())
    return logits,targets
        
        
def compute_metric(logits,targets,threshold=0.5):
    scores = [1 if i<threshold else 0 for i in logits]        
    accuracy = (np.array(targets)==np.array(scores))
    accuracy = sum(accuracy)/len(accuracy)
    f1 = f1_score(targets,scores)
    print(accuracy)        
    print(f1_score(targets,scores))    
    return accuracy,f1



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenization",default='words',type=str)
    parser.add_argument('--hidden_dim',default=300,type=int)
    parser.add_argument('--out_dim',default=300,type=int)
    parser.add_argument("--training_data",default='Data/tr-samples_contrastive',type=str)
    parser.add_argument("--develop_data",default='Data/de-samples_contrastive',type=str)
    parser.add_argument("--test_data",default='Data/te-samples_contrastive',type=str)
    parser.add_argument("--train",default=True,type=bool)
    parser.add_argument("--test",default=True,type=bool)
    parser.add_argument("--save_model",default='./model',type=str)
    parser.add_argument("--load_model",default=None,type=str)
    parser.add_argument("--distance",default='cosine',type=str)
    parser.add_argument("--tau_low",default=0.2,type=float)
    parser.add_argument("--tau_high",default=0.8,type=float)
    parser.add_argument("--epochs",default=10,type=int)
    parser.add_argument("--grad_acc",default=1,type=int)
    parser.add_argument("--lr",default=1e-3,type=float)
    parser.add_argument('--training_bsz',default=64,type=int)
    parser.add_argument('--test_bsz',default=128,type=int)
    parser.add_argument('--save_epochs',default=2,type=int)
    args = parser.parse_args()
    
    
    
    device = "cuda"
    #initiate the tokenizer
    if args.tokenization == 'bert-base-uncased':
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        vocab = len(tokenizer)
    elif args.tokenization == 'bert-base-cased':
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
        vocab = len(tokenizer)
    elif args.tokenization == 'roberta-base':
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        vocab = len(tokenizer)
    elif args.tokenization == 'words':
        tokenizer = word_tokenize
        word_dict = {}
        with open('vocab_30000.txt','r') as f:
            for line in f.readlines():
                k,v = line.strip().split('\t\t')
                word_dict[k] = int(v)
        vocab = len(word_dict)
        
    
    #initiate the models
    if args.load_model:
        model = torch.load(args.load_model)
    else:
        model = RNN(hidden_dim=args.hidden_dim,vocab=vocab,out_dim=args.out_dim).to(device)
    
    
    optimizer = AdamW(model.parameters(),lr=args.lr)
    loss_fn = contrastive_loss
    

#    lambda2 = lambda epoch: 0.99 ** epoch
#    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda2)
    
    threshold = 0.5*args.tau_low+0.5*args.tau_high
    model_name = "lstm-%s-%s-%s-tokenize-%s"%(args.distance,args.tau_low,args.tau_high,args.tokenization)
    if not os.path.exists(os.path.join(args.save_model,model_name)):
        os.mkdir(os.path.join(args.save_model,model_name))
        
        
        
    if args.train:
        # load data
        text_tr, label_tr,_,_ = loads(args.training_data)
        text_te, label_te,_,_ = loads(args.develop_data)
        
        text_tr, label_tr = shuffle(text_tr, label_tr)
        text_te, label_te = shuffle(text_te, label_te)
    
        # preprocess the data
        trainA, trainB = preprocess(text_tr)
        testA, testB = preprocess(text_te)
        
        # tokenize the data
        if args.tokenization != 'words':
            train_a, train_b, tr_attn_a, tr_attn_b = tokenize(trainA, trainB)
            test_a, test_b, te_attn_a, te_attn_b = tokenize(testA, testB)
        else:
            train_a, train_b, tr_attn_a, tr_attn_b = tokenize_word(trainA, trainB)
            test_a, test_b, te_attn_a, te_attn_b = tokenize_word(testA, testB)
        

        for k in range(1,args.epochs+1):
            with open(os.path.join(args.save_model,model_name,'results'),'a+') as out:
                train(args,model,train_a,train_b,tr_attn_a, tr_attn_b, label_tr)
                logits,targets = evaluate(args,test_a,test_b,te_attn_a, te_attn_b, label_te)
                accuracy,f1 = compute_metric(logits,targets,threshold=threshold)
                out.write("Epoach:%s-Acc:%s-F1:%s\n"%(k,accuracy,f1))
                if k%args.save_epochs==0:
                    torch.save(model,os.path.join(args.save_model,model_name,'model-%s'%k))
            
    
    if args.test:
        # load data
        text_te, label_te, users, products = loads(args.test_data)
#        text_te, label_te = shuffle(text_te, label_te)
        testA, testB = preprocess(text_te)
        if args.tokenization != 'words':
            test_a, test_b, te_attn_a, te_attn_b = tokenize(testA, testB)
        else:
            test_a, test_b, te_attn_a, te_attn_b = tokenize_word(testA, testB)
        
        logits,targets = evaluate(args,test_a,test_b,te_attn_a, te_attn_b, label_te)
        accuracy,f1 = compute_metric(logits,targets,threshold=threshold)

        with open(os.path.join(args.save_model,model_name,'full_evaluation'),'w') as out:
            for u, p, logit, target in zip(users,products,logits,targets):
                out.write("%s\t\t%s\t\t%s\t\t%s\n"%(u,p,logit,target))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    