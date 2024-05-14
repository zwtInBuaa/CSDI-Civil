import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
import pandas as pd


def get_quantile(samples, q, dim=1):
    return torch.quantile(samples, q, dim=dim).cpu().numpy()


dataset = 'Civil'  # dataset to choose

nsample = 100  # number of generated sample

path = './save/pm25_validationindex0_20240513_202142/generated_outputs_nsample100.pk'
with open(path, 'rb') as f:
    samples, all_target, all_evalpoint, all_observed, all_observed_time, scaler, mean_scaler = pickle.load(f)

path_1 = './save/pm25_validationindex0_20240501_081738/generated_outputs_nsample100.pk'
with open(path_1, 'rb') as f:
    samples_1, all_target_1, all_evalpoint_1, all_observed_1, all_observed_time_1, scaler_1, mean_scaler_1 = pickle.load(
        f)

all_target_np = all_target.cpu().numpy()
all_evalpoint_np = all_evalpoint.cpu().numpy()
all_observed_np = all_observed.cpu().numpy()
all_given_np = all_observed_np - all_evalpoint_np

K = samples.shape[-1]  # feature
L = samples.shape[-2]  # time length

if dataset == 'Civil':
    path = './data/ours/our_meanstd.pk'
    with open(path, 'rb') as f:
        train_mean, train_std = pickle.load(f)
    train_std_cuda = torch.Tensor(train_std).cuda()
    train_mean_cuda = torch.Tensor(train_mean).cuda()
    all_target_np = (all_target_np * train_std + train_mean)
    samples = (samples * train_std_cuda + train_mean_cuda)
    samples_1 = (samples_1 * train_std_cuda + train_mean_cuda)

qlist = [0.05, 0.25, 0.5, 0.75, 0.95]
quantiles_imp = []
quantiles_imp_1 = []
for q in qlist:
    quantiles_imp.append(get_quantile(samples, q, dim=1) * (1 - all_given_np) + all_target_np * all_given_np)
    quantiles_imp_1.append(get_quantile(samples_1, q, dim=1) * (1 - all_given_np) + all_target_np * all_given_np)

dataind = 0  # change to visualize a different sample

# plt.rcParams["font.size"] = 16

for dataind in range(6):
    fig, axes = plt.subplots(nrows=18, ncols=4, figsize=(24, 72))
    fig.delaxes(axes[-1][-1])
    for k in range(K):
        df = pd.DataFrame(
            {"x": np.arange(0, L), "val": all_target_np[dataind, :, k], "y": all_evalpoint_np[dataind, :, k]})
        df = df[df.y != 0]
        df2 = pd.DataFrame(
            {"x": np.arange(0, L), "val": all_target_np[dataind, :, k], "y": all_given_np[dataind, :, k]})
        df2 = df2[df2.y != 0]
        row = k // 4
        col = k % 4
        axes[row][col].plot(range(0, L), quantiles_imp[2][dataind, :, k], color='g', linestyle='solid', label='CSDI')
        axes[row][col].fill_between(range(0, L), quantiles_imp[0][dataind, :, k], quantiles_imp[4][dataind, :, k],
                                    color='g', alpha=0.3)

        axes[row][col].plot(range(0, L), quantiles_imp_1[2][dataind, :, k], color='gray', linestyle='solid',
                            label='CSDI')
        axes[row][col].fill_between(range(0, L), quantiles_imp_1[0][dataind, :, k], quantiles_imp_1[4][dataind, :, k],
                                    color='gray', alpha=0.3)

        axes[row][col].plot(df.x, df.val, color='b', marker='o', linestyle='None')
        axes[row][col].plot(df2.x, df2.val, color='r', marker='x', linestyle='None')
        if col == 0:
            plt.setp(axes[row, 0], ylabel='value')
        if row == -1:
            plt.setp(axes[-1, col], xlabel='time')

    plt.savefig("res" + str(dataind) + '.png')
    plt.close()
