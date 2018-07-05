# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 10:54:03 2018

@author: achyu
"""

import pandas as pd
df=pd.read_csv('Ads_CTR_Optimisation.csv').values
N=10000
d=10
no_of_sel=[0]*d
sum_of_rew=[0]*d
ads_sel=[]
tot_rew=0
import math
for n in range(0,N):
    max_ub=0
    ad=0
    for i in range(0,d):
        if no_of_sel[i]>0:
            avg_rew=sum_of_rew[i]/no_of_sel[i]
            delta=math.sqrt(3/2 * math.log(n+1)/ no_of_sel[i])
            ub=avg_rew+delta
        else:
            ub=1e400
        if ub>max_ub:
            max_ub=ub
            ad=i
    ads_sel.append(ad)
    no_of_sel[ad]+=1
    sum_of_rew[ad]+=df[n,ad]
    tot_rew+=df[n,ad]
        
import matplotlib.pyplot as plt
plt.hist(ads_sel)
plt.xlabel('ads')
plt.ylabel('times selected')
plt.show()
        