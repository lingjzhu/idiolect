#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 17:06:21 2020

"""

import os
import pandas as pd
import torch
import transformers
import numpy as np
import argparse

from torch import nn
import torch.nn.functional as F
#from transformers.modeling_roberta import RobertaPreTrainedModel
from transformers import RobertaConfig, RobertaModel, AdamW,RobertaTokenizerFast
from sklearn.utils import shuffle
from tqdm import tqdm
from sklearn.metrics import f1_score,roc_auc_score
#from entmax import sparsemax


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



def tokenize(textA,textB,seq_len=102):
    
    length = len(textA)
    input_ids_a = torch.ones(length,seq_len).long()
    input_ids_b = torch.ones(length,seq_len).long()
    attention_mask_a = torch.zeros(length,seq_len).long()
    attention_mask_b = torch.zeros(length,seq_len).long()
    
    for k, a, b in tqdm(zip(range(length),textA, textB)):
        seqA = tokenizer.encode_plus(a,add_special_tokens=True, max_length=seq_len,truncation=True,return_tensors="pt")
        seqB = tokenizer.encode_plus(b,add_special_tokens=True, max_length=seq_len,truncation=True,return_tensors="pt")
        if args.permute_words==True:
            seqA['input_ids'] = seqA['input_ids'][:,torch.randperm(seqA['input_ids'].size(1))]
            seqB['input_ids'] = seqB['input_ids'][:,torch.randperm(seqB['input_ids'].size(1))]
        input_ids_a[k,:seqA['input_ids'].shape[1]] = seqA['input_ids']
        input_ids_b[k,:seqB['input_ids'].shape[1]] = seqB['input_ids']
        attention_mask_a[k,:seqA['attention_mask'].shape[1]] = seqA['attention_mask']
        attention_mask_b[k,:seqB['attention_mask'].shape[1]] = seqB['attention_mask']
    return input_ids_a,input_ids_b,attention_mask_a,attention_mask_b


def batchify(text_a,text_b,attn_a,attn_b,labels,size=16,shuffling=False):
    length = len(text_a)
    
    if shuffling == True:
        text_a,text_b,attn_a,attn_b,labels = shuffle(text_a,text_b,attn_a,attn_b,labels)
        
    for i in range(0,length,size):
        sampleA = text_a[i:min(i+size,length)]
        sampleB = text_b[i:min(i+size,length)]
        sample_attn_a = attn_a[i:min(i+size,length)]
        sample_attn_b = attn_b[i:min(i+size,length)]
        y = labels[i:min(i+size,length)]

        yield sampleA, sampleB, sample_attn_a, sample_attn_b, torch.tensor(y).long()



def masking(text,mlm_prob=0.1,mask=50264):
    
    indices_replaced = torch.bernoulli(torch.full(text.shape, mlm_prob)).bool()
    text[indices_replaced] = mask
    return text
    

def margin_loss(diff, target, tau_low=0.2, tau_high=0.8):
    
    within = target*torch.max(diff-tau_low,torch.tensor(0.0).to(device))**2
    without = (1-target)*torch.max(tau_high-diff,torch.tensor(0.0).to(device))**2
    
    return torch.mean(within)+torch.mean(without)



def proxy_anchor_loss(cosine,target,delta=0.5,alpha=10):
    '''
    Implementation of the loss in:
        https://openaccess.thecvf.com/content_CVPR_2020/papers/Kim_Proxy_Anchor_Loss_for_Deep_Metric_Learning_CVPR_2020_paper.pdf

    '''
    positive = cosine[target==1]
    negative = cosine[target!=1]
    pos_loss = torch.mean(F.softplus(torch.logsumexp(-alpha*(positive.unsqueeze(1) - delta),dim=1)))
    neg_loss = torch.mean(F.softplus(torch.logsumexp(alpha*(negative.unsqueeze(1) + delta),dim=1)))
    
    
    if torch.isnan(pos_loss):
        return neg_loss
    elif torch.isnan(neg_loss):
        return pos_loss
    else:
        return pos_loss+neg_loss
    

def modified_anchor_loss(cosine,target,delta_s=0.8,delta_d=0.1,alpha=10):

    positive = cosine[target==1]
    negative = cosine[target!=1]
    pos_loss = torch.mean(F.softplus(torch.logsumexp(-alpha*(positive.unsqueeze(1) - delta_s),dim=1)))
    neg_loss = torch.mean(F.softplus(torch.logsumexp(alpha*(negative.unsqueeze(1) - delta_d),dim=1)))
    
    
    if torch.isnan(pos_loss):
        return neg_loss
    elif torch.isnan(neg_loss):
        return pos_loss
    else:
        return pos_loss+neg_loss


def margin_anchor_loss(cosine,target,delta_s=0.8,delta_d=0.2,alpha=10):

    positive = cosine[target==1]
    negative = cosine[target!=1]
    
    pos_loss = torch.mean(F.softplus(torch.logsumexp(alpha*torch.max(delta_s - positive.unsqueeze(1),torch.tensor(0.0).to(device)),dim=1)))
    neg_loss = torch.mean(F.softplus(torch.logsumexp(alpha*torch.max(negative.unsqueeze(1) - delta_d,torch.tensor(0.0).to(device)),dim=1)))
    
    if torch.isnan(pos_loss):
        return neg_loss
    elif torch.isnan(neg_loss):
        return pos_loss
    else:
        return pos_loss+neg_loss


def spherical_regularizer(embeddings,eta=0.5):
    '''
    Implementation of Eq (17) in 
    https://proceedings.neurips.cc/paper/2020/file/d9812f756d0df06c7381945d2e2c7d4b-Paper.pdf
    '''    
    
    norms = torch.norm(embeddings,dim=1)
    norms = norms - torch.mean(norms)
    return eta*torch.mean(norms)



class AttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        self.softmax = nn.functional.softmax

    def forward(self, batch_rep, att_mask=None):
        """
            N: batch size, T: sequence length, H: Hidden dimension
            input:
                batch_rep : size (N, T, H)
            attention_weight:
                att_w : size (N, T, 1)
            return:
                utter_rep: size (N, H)
        """
        att_logits = self.W(batch_rep).squeeze(-1)
        if att_mask is not None:
            att_logits = att_mask + att_logits
        att_w = self.softmax(att_logits, dim=-1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep


class DNNSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_dim,
        **kwargs
    ):
        super(DNNSelfAttention, self).__init__()
        self.pooling = AttentionPooling(hidden_dim)
        self.out_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, features, att_mask):
        out = self.pooling(features, att_mask).squeeze(-1)
        predicted = self.out_layer(out)
        return predicted


class SRoberta(nn.Module):
    
    def __init__(self,model_name):
        super().__init__()
        
        self.roberta = RobertaModel.from_pretrained(model_name,return_dict=True)
        self.pooler = DNNSelfAttention(768)
        
    def forward(self,input_ids,att_mask=None):
        out = self.roberta(input_ids,att_mask)
        out = out.last_hidden_state
        out = self.pooler(out,att_mask)
        return out


class SRoberta_vanilla(nn.Module):

    def __init__(self,model_name):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name,return_dict=True)
        
    def forward(self, inut_ids,att_mask):
        out = self.roberta(input_ids,att_mask)
        out = out.last_hidden_state
        return out
 

def train(args,model,train_a,train_b,tr_attn_a, tr_attn_b, label_tr):

    model.train()
    for i, (pair_a, pair_b, attna, attnb, y) in tqdm(enumerate(batchify(train_a,train_b,tr_attn_a, tr_attn_b, label_tr,size=args.training_bsz,shuffling=True))):
        
        if args.mask_prob!=0:
            pair_a = masking(pair_a,mlm_prob=args.mask_prob)
            pair_b = masking(pair_b,mlm_prob=args.mask_prob)
        
        hiddenA = model(input_ids=pair_a.to(device),att_mask=attna.to(device))
        hiddenB = model(input_ids=pair_b.to(device),att_mask=attnb.to(device))
        
        
        if args.spherical:
            reg = 0.5*spherical_regularizer(hiddenA,eta=args.eta) + 0.5*spherical_regularizer(hiddenB,eta=args.eta)
        
        if args.loss == 'anchor':
            hiddenA = F.normalize(hiddenA,dim=1)
            hiddenB = F.normalize(hiddenB,dim=1)
            cosine = torch.sum(hiddenA*hiddenB,dim=1)
            loss = 0.1*loss_fn(cosine,y.to(device).float(),delta=args.delta,alpha=args.alpha)
            
        elif args.loss == 'modified_anchor' or args.loss == 'margin_anchor':
            hiddenA = F.normalize(hiddenA,dim=1)
            hiddenB = F.normalize(hiddenB,dim=1)
            cosine = torch.sum(hiddenA*hiddenB,dim=1)
            loss = 0.1*loss_fn(cosine,y.to(device).float(),delta_s=args.tau_high,delta_d=args.tau_low,alpha=args.alpha)
            
        elif args.loss == 'margin':
            if args.distance == 'cosine':
                hiddenA = F.normalize(hiddenA,dim=1)
                hiddenB = F.normalize(hiddenB,dim=1)
                cosine = 1.0-torch.sum(hiddenA*hiddenB,dim=1)
                loss = loss_fn(cosine,y.to(device).float(),tau_low=args.tau_low,tau_high=args.tau_high)
            elif args.distance == 'euclidean':
                cosine = torch.norm(hiddenA - hiddenB,dim=1)
                loss = loss_fn(cosine,y.to(device).float(),tau_low=args.tau_low,tau_high=args.tau_high)
            
        if args.spherical:
            loss = loss + reg
        
        loss.backward()
        if i%args.grad_acc==0:
            optimizer.step()
            optimizer.zero_grad()
            print("The loss is: %s"%loss.item())
        
        
                

def evaluate(args,test_a,test_b,te_attn_a, te_attn_b, label_te):
    #evaluate
    logits = []
    targets = []
    model.eval()
    for i, (pair_a, pair_b, attna, attnb, y) in tqdm(enumerate(batchify(test_a,test_b,te_attn_a, te_attn_b, label_te,size=args.test_bsz))):

        with torch.no_grad():
            hiddenA = model(pair_a.to(device),att_mask=attna.to(device))
            hiddenB = model(pair_b.to(device),att_mask=attnb.to(device))

            
            if args.distance == 'cosine':
                hiddenA = F.normalize(hiddenA,dim=1)
                hiddenB = F.normalize(hiddenB,dim=1)
                logit = 1.0-torch.sum(hiddenA*hiddenB,dim=1)
            elif args.distance == 'euclidean':
                logit = torch.norm(hiddenA - hiddenB,dim=1)
                
            logits += list(logit.detach().cpu().numpy())
            targets += list(y.detach().cpu().numpy())
    return logits,targets
        
        
def compute_metric(logits,targets,threshold):
    scores = [1 if i<threshold else 0 for i in logits]        
    accuracy = (np.array(targets)==np.array(scores))
    accuracy = sum(accuracy)/len(accuracy)
    f1 = f1_score(targets,scores)
    auc = roc_auc_score(targets,1-np.array(logits))
    print(accuracy)        
    print(f1_score(targets,scores)) 
    print(auc)
    return round(accuracy,3),round(f1,3),round(auc,3)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",default='roberta-base',type=str)
    parser.add_argument("--attpool",default=True,type=bool)
    parser.add_argument("--loss", default='anchor',type=str)
    parser.add_argument("--training_data",default=None,type=str)
    parser.add_argument("--develop_data",default=None,type=str)
    parser.add_argument("--test_data",default=None,type=str)
    parser.add_argument("--train",action='store_true')
    parser.add_argument("--test",action='store_true')
    parser.add_argument("--save_model",default='/gpfs/accounts/lingjzhu_root/lingjzhu1/lingjzhu/authorship_models',type=str)
    parser.add_argument("--load_model",default=None,type=str)
    parser.add_argument("--save_cache",default=None,type=str)
    parser.add_argument("--load_cache",default=None,type=str)
    parser.add_argument("--distance",default='cosine',type=str)
    parser.add_argument("--tau_low",default=0.2,type=float)
    parser.add_argument("--tau_high",default=0.8,type=float)
    parser.add_argument("--epochs",default=3,type=int)
    parser.add_argument("--grad_acc",default=8,type=int)
    parser.add_argument("--mask_prob",default=0.1,type=float)
    parser.add_argument("--lr",default=1e-5,type=float)
    parser.add_argument('--training_bsz',default=32,type=int)
    parser.add_argument('--test_bsz',default=80,type=int)
    parser.add_argument('--save_test_data',action='store_true')
    parser.add_argument('--permute_words',action='store_true')
    parser.add_argument('--alpha',default=10,type=float)
    parser.add_argument('--delta',default=0.5,type=float)
    parser.add_argument('--spherical',action='store_true')
    parser.add_argument('--eta', default=0.5, type=float)
    parser.add_argument('--prefix',default=None,type=str)
    args = parser.parse_args()
    
    
    
    device = "cuda"
    #initiate the tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained(args.model)
    #initiate the models
    if args.load_model:
        model = torch.load(args.load_model)
    else:
        if args.attpool == True:
            model = SRoberta('roberta-base').to(device)
        else:
            model = SRoberta_vanilla('roberta-base').to(device)
    optimizer = AdamW(model.parameters(),lr=args.lr)
    
    if args.loss == 'margin':
        loss_fn = margin_loss
        threshold = 0.5*args.tau_low+0.5*args.tau_high
    elif args.loss == 'anchor':
        loss_fn = proxy_anchor_loss
        threshold = args.delta
    elif args.loss == 'modified_anchor':
        loss_fn = modified_anchor_loss
        threshold = 0.5*args.tau_low+0.5*args.tau_high
    elif args.loss == 'margin_anchor':
        loss_fn = margin_anchor_loss
        threshold = 0.5*args.tau_low+0.5*args.tau_high        
        
    
    
    
    if args.loss == 'margin':
        model_name = "roberta-%s-%s-%s-mask-%s"%(args.distance,args.tau_low,args.tau_high,args.mask_prob)
    elif args.loss == 'anchor':
        model_name = "roberta-%s-%s-mask-%s-delta-%s-alpha-%s"%(args.distance,args.loss,args.mask_prob,args.delta,args.alpha)
    elif args.loss == 'modified_anchor':
        model_name = "roberta-%s-%s-mask-%s-delta-%s-%s-alpha-%s"%(args.distance,args.loss,args.mask_prob,args.tau_low,args.tau_high,args.alpha)
    elif args.loss == 'margin_anchor':
        model_name = "roberta-%s-%s-mask-%s-delta-%s-%s-alpha-%s"%(args.distance,args.loss,args.mask_prob,args.tau_low,args.tau_high,args.alpha)
        
    if args.spherical:
        model_name += "-sp-eta-%s"%(args.eta)

    if args.attpool != True:
        model_name = 'vallina-' + model_name
        
    if args.prefix:
        model_name = args.prefix + model_name
    
    if not os.path.exists(os.path.join(args.save_model,model_name)):
        os.mkdir(os.path.join(args.save_model,model_name))
        
        
        
    if args.train:
        # load data
        if args.save_cache:
            text_tr, label_tr,_,_ = loads(args.training_data)
            text_te, label_te,_,_ = loads(args.develop_data)
            
            text_tr, label_tr = shuffle(text_tr, label_tr)
            text_te, label_te = shuffle(text_te, label_te)
        
            # preprocess the data
            trainA, trainB = preprocess(text_tr)
            testA, testB = preprocess(text_te)
            
            # tokenize the data
            train_a, train_b, tr_attn_a, tr_attn_b = tokenize(trainA, trainB)
            test_a, test_b, te_attn_a, te_attn_b = tokenize(testA, testB)
            
            torch.save((train_a, train_b, tr_attn_a, tr_attn_b,label_tr),os.path.join(args.save_cache,'train'))
            torch.save((test_a, test_b, te_attn_a, te_attn_b,label_te),os.path.join(args.save_cache,'dev'))
            
        elif args.load_cache:
            train_a, train_b, tr_attn_a, tr_attn_b, label_tr = torch.load(os.path.join(args.load_cache,'train'))
            test_a, test_b, te_attn_a, te_attn_b, label_te = torch.load(os.path.join(args.load_cache,'dev'))
            
        

        for k in range(args.epochs):
            with open(os.path.join(args.save_model,model_name,'results'),'a+') as out:
                train(args,model,train_a,train_b,tr_attn_a, tr_attn_b, label_tr)
                logits,targets = evaluate(args,test_a,test_b,te_attn_a, te_attn_b, label_te)
                accuracy,f1,auc = compute_metric(logits,targets,threshold=threshold)
                torch.save(model,os.path.join(args.save_model,model_name,'model-%s'%k))
                out.write("Epoach:%s-Acc:%s-F1:%s-AUC:%s\n"%(k,accuracy,f1,auc))
            
    
    if args.test:
        # load data
        text_te, label_te, users, products = loads(args.test_data)
#        text_te, label_te = shuffle(text_te, label_te)
        if args.save_cache:
            testA, testB = preprocess(text_te)
            test_a, test_b, te_attn_a, te_attn_b = tokenize(testA, testB)
            torch.save((test_a, test_b, te_attn_a, te_attn_b),os.path.join(args.save_cache,'test'))
        elif args.load_cache:
            test_a, test_b, te_attn_a, te_attn_b = torch.load(os.path.join(args.load_cache,'test'))
        
        logits,targets = evaluate(args,test_a,test_b,te_attn_a, te_attn_b, label_te)
        accuracy,f1,auc = compute_metric(logits,targets,threshold=threshold)
        with open(os.path.join(args.save_model,model_name,'results'),'a+') as out:
            out.write('Test-%s\n'%(args.test_data))
            out.write("Acc:%s-F1:%s-AUC:%s\n"%(accuracy,f1,auc))
        if args.save_test_data:
            with open(os.path.join(args.save_model,model_name,'full_evaluation'),'w') as out:
                for u, p, logit, target in zip(users,products,logits,targets):
                    out.write("%s\t\t%s\t\t%s\t\t%s\n"%(u,p,logit,target))                                      
    
    
    
      

    
    
    
    
    
    
    
    
    
    
    
    
