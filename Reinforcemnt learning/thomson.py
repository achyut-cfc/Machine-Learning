# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 12:19:20 2018

@author: achyu
"""

import pandas as pd
import random
df=pd.read_csv('Ads_CTR_Optimisation.csv').values
N=10000
d=10
ads_sel=[]
no_of_rew_1=[0]*d
no_of_rew_0=[0]*d
tot_rew=0
for n in range(0,N):
    max_random=0
    ad=0
    for i in range(0,d):
        random_beta=random.betavariate(no_of_rew_1[i]+1,no_of_rew_0[i]+1)
        if random_beta>max_random:
            max_random=random_beta
            ad=i
    ads_sel.append(ad)
    reward=df[n,ad]
    if reward==1:
        no_of_rew_1[ad]+=1
    else:
        no_of_rew_0[ad]+=1
    tot_rew+=reward
import matplotlib.pyplot as plt
plt.hist(ads_sel)
plt.xlabel('ad')
plt.ylabel('times selected')