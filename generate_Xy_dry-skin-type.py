# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 16:03:56 2025

@author: Agrima Agarwal
"""
import numpy as np
import os
import pandas as pd
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

df = pd.read_excel('dry_skin_type.xlsx',header=None)  # Replace with your actual file path or name
black_list = ['S003','S006','S036'] #dry

D=[]
for j in tqdm(os.listdir('raw files dry skin')):
    if j not in black_list:
        Dd = patient_analysis('raw files dry skin/'+j)
        Dd.type = df.loc[df.iloc[:, 0] == j].iloc[0, 1]
        D.append(Dd)
print(len(D))

#%% Extract flattened impulses and labels

imp_all = np.array([(obj.roi.impulses).T.flatten() for obj in D])
labels = np.array([obj.type for obj in D])

X = imp_all[np.where((labels== 'p') | (labels == 'e'))]
y = labels[(labels == 'p') | (labels == 'e')]
y = np.where(y == 'p', 1, 0)

np.savez('Xy_dry-skin-type', X=X, y=y)

