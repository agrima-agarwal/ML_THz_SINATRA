# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 16:28:03 2025

@author: Agrima Agarwal
"""
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
from sklearn.decomposition import PCA



plt.rcdefaults()
# === Set Global Plot Style ===
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'legend.fontsize': 14,
    'lines.linewidth': 3,
    'axes.linewidth': 1.5,
    'savefig.dpi': 300
})

data = np.load('Xy_dry-skin-no-moist.npz')
X = data['X']
y = data['y']

#%% zoomed in impulse for different classes

# === PCA Plot ===
plt.figure(figsize=(6, 8))

plt.plot(np.mean(X[37:,:],axis=0),linestyle='--', color='blue', label='Healthy')
plt.plot(np.mean(X[:37,:],axis=0),linestyle='-', color='red', label='Dry')

# plt.plot(np.mean(X[y==0,:],axis=0),linestyle='--', color='blue', label='Eczema')
# plt.plot(np.mean(X[y==1,:],axis=0),linestyle='-', color='red', label='Psoriasis')

# plt.plot(np.mean(X[37:,:],axis=0),linestyle='--', color='blue', label='Healthy')
# plt.plot(np.mean(X[:37,:],axis=0),linestyle='-', color='red', label='Skin cancer')

plt.xlim([8000, 8400])
plt.xlabel('Data points')
plt.ylabel('Amplitude')
plt.legend(loc='upper right',fontsize=18)
plt.ylim([-0.027,0.034])
# plt.grid()
plt.tight_layout()
plt.show()

#%% Dimensionality reduction using PCA

pca = PCA(n_components=2) 
pca.fit(X)
transformed_data = pca.transform(X)

plt.figure(figsize=(6, 8))
plt.plot(pca.mean_, color='black', label='Mean')
plt.plot(pca.components_[0, :].T, linestyle='--', color='green', label='P.C. 1')
plt.plot(pca.components_[1, :].T, linestyle='-.', color='deeppink', label='P.C. 2')

plt.xlim([8000, 8400])
plt.xlabel('Data points')
plt.ylabel('Characteristic Response')
plt.legend(loc='upper right',fontsize=18)
plt.ylim([-0.025,0.027])
# plt.grid()
plt.tight_layout()
plt.show()

#%% box plots of P.C. 1 score for different classes
 
data = [transformed_data[y==0,0], transformed_data[y==1,0]]
positions = [1, 1.5]
plt.figure(figsize=(5.5, 6))

bp = plt.boxplot(
    data,
    positions=positions,
    widths=0.2,
    labels=['Healthy', 'Dry'],
    # labels=['Eczema', 'Psoriasis'],
    # labels=['Healthy', 'Skin cancer'],
    patch_artist=True,
    boxprops=dict(linewidth=4),       # make box edges bold
    whiskerprops=dict(linewidth=4),   # make whiskers bold
    capprops=dict(linewidth=4),       # make caps bold
    medianprops=dict(linewidth=4, color="black")  # bold red median line
)

colors = ['lightskyblue', 'lightcoral']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

plt.ylabel('Score of P.C. 1')
plt.xticks(fontsize=16)  
plt.tight_layout()
plt.show()

