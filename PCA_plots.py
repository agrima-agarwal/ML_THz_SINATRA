# -*- coding: utf-8 -*-

"""

Runs the SAME plotting/PCA workflow for all five THz classification datasets
in one script.

"""

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import ttest_ind
import os

PCA_OUTPUT_DIR = 'outputs/pca'
os.makedirs(PCA_OUTPUT_DIR, exist_ok=True)


plt.rcdefaults()

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


TASKS = [
    {
        'name': 'Dry vs Healthy - no moisturizer',
        'file': 'outputs/Xy/Xy_dry-skin-no-moist.npz',
        'class0': 'Healthy',
        'class1': 'Dry',
    },
    {
        'name': 'Dry vs Healthy - 10 min moisturizer',
        'file': 'outputs/Xy/Xy_dry-skin-10-moist.npz',
        'class0': 'Healthy',
        'class1': 'Dry',
    },
    {
        'name': 'Dry vs Healthy - 20 min moisturizer',
        'file': 'outputs/Xy/Xy_dry-skin-20-moist.npz',
        'class0': 'Healthy',
        'class1': 'Dry',
    },
    {
        'name': 'Eczema vs Psoriasis',
        'file': 'outputs/Xy/Xy_dry-skin-type.npz',
        'class0': 'Eczema',
        'class1': 'Psoriasis',
    },
    {
        'name': 'Skin Cancer vs Healthy',
        'file': 'outputs/Xy/Xy_skin-cancer.npz',
        'class0': 'Healthy',
        'class1': 'Skin cancer',
    },
]


# Generate the class-comparison and PCA plots for one task.
def make_plots(task):
    task_name = (
    task['name']
    .lower()
    .replace(' ', '_')
    .replace('/', '_')
    .replace('-', '_')
    )

    print('\n' + '=' * 80)
    print(task['name'])
    print('=' * 80)

    data = np.load(task['file'])
    X = data['X']
    y = data['y']

    class0 = task['class0']
    class1 = task['class1']


    plt.figure(figsize=(6, 8))

    x_axis = np.arange(X.shape[1])

    mean0 = np.mean(X[y == 0, :], axis=0)
    std0 = np.std(X[y == 0, :], axis=0, ddof=1)

    mean1 = np.mean(X[y == 1, :], axis=0)
    std1 = np.std(X[y == 1, :], axis=0, ddof=1)

    plt.plot(
        x_axis,
        mean0,
        linestyle='--',
        color='blue',
        label=class0
    )

    # plt.fill_between(
    #     x_axis,
    #     mean0 - std0,
    #     mean0 + std0,
    #     color='blue',
    #     alpha=0.15
    # )

    plt.plot(
        x_axis,
        mean1,
        linestyle='-',
        color='red',
        label=class1
    )

    # plt.fill_between(
    #     x_axis,
    #     mean1 - std1,
    #     mean1 + std1,
    #     color='red',
    #     alpha=0.15
    # )

    plt.xlim([8000, 8400])
    plt.xlabel('Data points')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right', fontsize=18, framealpha=0.5)    
    # plt.ylim([-0.035, 0.048]) #for std
    plt.ylim([-0.028, 0.034]) 
    

    plt.tight_layout()
    plt.savefig(
        os.path.join(PCA_OUTPUT_DIR, f'{task_name}_mean_impulse.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.show()


    pca = PCA(n_components=2)
    pca.fit(X)

    transformed_data = pca.transform(X)

    plt.figure(figsize=(6, 8))

    plt.plot(
        pca.mean_,
        color='black',
        label='Mean'
    )

    plt.plot(
        pca.components_[0, :].T,
        linestyle='--',
        color='green',
        label='P.C. 1'
    )

    plt.plot(
        pca.components_[1, :].T,
        linestyle='-.',
        color='deeppink',
        label='P.C. 2'
    )

    plt.xlim([8000, 8400])
    plt.xlabel('Data points')
    plt.ylabel('Characteristic Response')
    plt.legend(loc='upper right', fontsize=18, framealpha=0.5)    
    plt.ylim([-0.025, 0.03])


    plt.tight_layout()
    plt.savefig(
        os.path.join(PCA_OUTPUT_DIR, f'{task_name}_pca_components.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.show()


    data_box = [
        transformed_data[y == 0, 0],
        transformed_data[y == 1, 0]
    ]

    positions = [1, 1.5]

    plt.figure(figsize=(5.5, 6))

    bp = plt.boxplot(
        data_box,
        positions=positions,
        widths=0.2,
        labels=[class0, class1],
        patch_artist=True,
        boxprops=dict(linewidth=4),
        whiskerprops=dict(linewidth=4),
        capprops=dict(linewidth=4),
        medianprops=dict(linewidth=4, color="black")
    )

    colors = ['lightskyblue', 'lightcoral']

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    plt.ylabel('Score of P.C. 1')
    plt.xticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(
        os.path.join(PCA_OUTPUT_DIR, f'{task_name}_pc1_boxplot.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.show()


    t_stat, p_ttest = ttest_ind(
        transformed_data[y == 0, 0],
        transformed_data[y == 1, 0],
        equal_var=False
    )

    print(
        f"Welch’s t-test:      "
        f"statistic = {t_stat:.3f}, "
        f"p-value = {p_ttest:.4f}"
    )

    stats_file = os.path.join(PCA_OUTPUT_DIR, 'PCA_statistics.txt')

    with open(stats_file, 'a', encoding='utf-8') as f:
        f.write(f"{task['name']}\n")
        f.write(f"PC1 explained variance = {pca.explained_variance_ratio_[0]:.6f}\n")
        f.write(f"PC2 explained variance = {pca.explained_variance_ratio_[1]:.6f}\n")
        f.write(f"{class0} PC1 mean = {np.mean(transformed_data[y == 0, 0]):.6f}\n")
        f.write(f"{class0} PC1 SD = {np.std(transformed_data[y == 0, 0], ddof=1):.6f}\n")
        f.write(f"{class1} PC1 mean = {np.mean(transformed_data[y == 1, 0]):.6f}\n")
        f.write(f"{class1} PC1 SD = {np.std(transformed_data[y == 1, 0], ddof=1):.6f}\n")
        f.write(f"Welch's t-test statistic = {t_stat:.6f}\n")
        f.write(f"Welch's t-test p-value = {p_ttest:.8g}\n\n")


# Run the script.
def main():

    stats_file = os.path.join(PCA_OUTPUT_DIR, 'PCA_statistics.txt')
    if os.path.exists(stats_file):
        os.remove(stats_file)

    for task in TASKS:
        make_plots(task)


if __name__ == '__main__':
    main()
