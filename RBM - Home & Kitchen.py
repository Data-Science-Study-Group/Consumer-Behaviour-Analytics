# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 12:46:59 2020

@author: juyee
"""

import pandas as pd
import numpy as np
from pandas import DataFrame 
import nltk
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

reviews = pd.read_json('Home_and_Kitchen_5.json',lines=True)
reviews2 = pd.read_json('beauty.json',lines=True)

reviews = reviews.append(reviews2,ignore_index=True,sort=True)

#Using the important columns
dfreviews = reviews[["reviewerID","asin","overall"]]
dfreviews = dfreviews.sort_values(by = 'reviewerID')
dfreviews.reset_index(drop=True,inplace=True)


count_asin = dfreviews.groupby("asin", as_index=False).count().rename(columns={"reviewerID":"totalReviewers"}).drop(columns=["overall"])
dfreviews = pd.merge(dfreviews, count_asin, how='right', on='asin')
dfreviews = dfreviews[dfreviews.totalReviewers >= 100]

count = dfreviews.groupby("reviewerID", as_index=False).count().rename(columns={"asin":"totalReviews"}).drop(columns=["totalReviewers","overall"])
dfreviews = pd.merge(dfreviews, count, how='right', on='reviewerID')
dfreviews = dfreviews[dfreviews.totalReviews >= 5]

unique_reviewers = dfreviews.reviewerID.unique()
unique_reviewers = pd.DataFrame(data=unique_reviewers)
unique_reviewers = unique_reviewers.rename(columns={0:'reviewerID'})
unique_reviewers.reset_index(drop=True,inplace=True)
unique_reviewers.index = np.arange(1,len(unique_reviewers)+1)
unique_reviewers['ind']=unique_reviewers.index


dfReviews = pd.merge(dfreviews,unique_reviewers,how='inner',on=['reviewerID'])

unique_asin = dfreviews.asin.unique()
unique_asin = pd.DataFrame(data=unique_asin)
unique_asin = unique_asin.rename(columns={0:'asin'})
unique_asin['asin_ind']=unique_asin.index
unique_asin.asin_ind = np.arange(1,len(unique_asin)+1)

dfReviews = pd.merge(dfReviews,unique_asin,how='inner',on=['asin'])



dfReviews['reviewerID'] = dfReviews['ind']
dfReviews['asin']=dfReviews['asin_ind']
dfReviews = dfReviews.drop(columns=["ind","asin_ind","totalReviews","totalReviewers"])
dfReviews.reset_index(drop=True,inplace=True)

#shortreviews = dfReviews[dfReviews['asin']<=12000][dfReviews['reviewerID']<=8000]

#dfReviews = dfReviews.sort_values(by = 'reviewerID')
#shortreviews = shortreviews.sort_values(by = 'reviewerID')

from sklearn.model_selection import train_test_split
training_set,test_set = train_test_split(dfReviews,test_size = 0.2,random_state=0)

training_set = training_set.sort_values(by = 'reviewerID')
training_set.reset_index(drop=True,inplace=True)
test_set = test_set.sort_values(by = 'reviewerID')
test_set.reset_index(drop=True,inplace=True)

#Making a copy of test set dataframe for backtracking
test_set_df = test_set

training_set = np.array(training_set,dtype="int")
test_set = np.array(test_set,dtype="int")

#test_set_temp = test_set

#nb_users = len(dfreviews.reviewerID.value_counts())
nb_users = len(dfReviews.reviewerID.value_counts())
nb_products = len(dfReviews.asin.value_counts())


#Converting data into an array with users in lines and movies in columns
def convert(data):
    new_data=[]
    for id_users in range(1,nb_users+1):
        id_products = data[:,1][data[:,0]==id_users]
        id_ratings = data[:,2][data[:,0]==id_users]
        ratings = np.zeros(nb_products)
        ratings[id_products-1]=id_ratings
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)



# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

#Converting the ratings into binary ratings 1(liked) and 0(not liked)
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1


test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

#Creating the architecture for neural network
class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv) #matrix
        self.a = torch.randn(1, nh) #bias for probability of hidden nodes given visible nodes
        self.b = torch.randn(1, nv) #bias for probability of visible nodes given hidden nodes
    #sampling hidden node according to probability p-h given v - sigmoid activation
    #x- visible neurons v in p_h_given_v
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v) #bernoulli used to sample probability of hidden node
    #if p_h_given_v for i'th hidden node is 0.7 and some random no is less than 0.7 then we will activate neuron otherwise we won't - explanation for bernoulli sampling
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    
    #contrastive divergence - minimizing energy by approximating gradients 
    #gibb sampling - sampling k times the hidden nodes and visible nodes - sample consecutively in chains - gibbs chain
    def train(self, v0, vk, ph0, phk):
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)
        
    def predict(self, x):# x: visible nodes
        _, h = self.sample_h( x)
        _, v = self.sample_v( h)
        return v

  
nv = len(training_set[0])
nh = 1200 #features that we want to detect
batch_size = 800
rbm = RBM(nv, nh)

#Training the RBM
nb_epoch = 15
for epoch in range(1,nb_epoch+1):
    train_loss = 0
    s = 0.
    for id_user in range(0,nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        #k steps of contrastive divergence - gibbs cycle
        for k in range(12):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0]-vk[v0>=0]))
        s+=1
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
    

import pickle
filename = 'rbm.sav'
pickle.dump(rbm,open(filename,'wb'))

pickle.dump(rbm,open('rbm.pkl','wb'))

rbm = pickle.load(open('rbm.pkl','rb'))



#Testing the RBM
test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1] #training set used to activate neurons of RBM
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1. #counter to normalize test loss
print('test loss: '+str(test_loss/s))

actual_set = test_set.numpy()
actual_trainset = training_set.numpy()
pred_set = rbm.predict(test_set).numpy()

#actual_train_set = training_set.numpy()
#pred_train_set = rbm.predict(training_set).numpy()

recommendations = pd.DataFrame(columns=["reviewerID","Products"])
for i in range(500):
    temp=[]
    for j in range(len(pred_set[i])):
        if pred_set[i][j]==1:
            prodid = "".join(unique_asin['asin'][unique_asin['asin_ind']==j+1].astype('str').tail(1).tolist())
            temp.append(prodid)
            
    reviewerid = "".join(unique_reviewers['reviewerID'][unique_reviewers['ind']==i+1].astype('str').tail(1).tolist())  
    
    recommendations = recommendations.append({'reviewerID':reviewerid,
                            'Products':temp
                           },ignore_index=True)
    
recommendations.to_csv("rbm_recommendations.csv",index=False)


recommendations.to_csv("test_recommendations.csv",index=False)




