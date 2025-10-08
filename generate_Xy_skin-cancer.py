# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 16:03:56 2025

@author: Agrima Agarwal
"""
import numpy as np
import os
from functions import THzData
from tqdm import tqdm

class patient_analysis:
    def __init__(self,folder,ROI = None):
        try:
            self.patient_id = folder
            roi = THzData(folder+'/roi',folder+'/reference.txt',folder+'/baseline.txt')
            con = THzData(folder+'/control',folder+'/reference.txt',folder+'/baseline.txt')
            self.roi = roi
            self.con = con
        except:
            print('no')          

#%% Do signal preprocessing for all measurements in dataset and store in D

black_list = ['S076']
D=[]
for j in tqdm(os.listdir('raw files skin cancer')):
    if j not in black_list:
        Dd = patient_analysis('raw files skin cancer/'+j)
        D.append(Dd)
print(len(D))

#%% Extract flattened impulses and labels

imp_all=[]
labels=[]
for i in ['roi','con']:
    imp_all1 = np.array([(getattr(obj, i).impulses).T.flatten() for obj in D]) 
    imp_all.extend(imp_all1)
    if i=='roi':
        lab = 1
    else:
        lab = 0
    label = np.repeat(lab, len(imp_all1))
    labels.extend(label)
    
imp_all = np.array(imp_all)
labels=np.array((labels),dtype=int)

X = imp_all[np.where((labels==0)|(labels==1))]
y = labels[(labels == 0) | (labels == 1)]

np.savez('Xy_skin-cancer', X=X, y=y)

