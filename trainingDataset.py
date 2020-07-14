#! python
# -*- coding: utf-8 -*-
# Author: kun
# @Time: 2019-07-23 14:36

from torch.utils.data.dataset import Dataset
import torch
import numpy as np


class trainingDataset(Dataset):
    def __init__(self, datasetA, datasetB, n_frames=128):
        self.datasetA = datasetA
        self.datasetB = datasetB
        self.n_frames = n_frames

    def __getitem__(self, index):
        dataset_A = self.datasetA
        dataset_B = self.datasetB
        n_frames = self.n_frames

        self.length = min(len(dataset_A), len(dataset_B))

        num_samples = min(len(dataset_A), len(dataset_B))
        train_data_A_idx = np.arange(len(dataset_A))
        train_data_B_idx = np.arange(len(dataset_B))
        np.random.shuffle(train_data_A_idx)
        np.random.shuffle(train_data_B_idx)
        train_data_A_idx_subset = train_data_A_idx[:num_samples]
        train_data_B_idx_subset = train_data_B_idx[:num_samples]

        train_data_A = list()
        train_data_B = list()

        for idx_A, idx_B in zip(train_data_A_idx_subset, train_data_B_idx_subset):
            data_A = dataset_A[idx_A]
            frames_A_total = data_A.shape[1]
            assert frames_A_total >= n_frames
            start_A = np.random.randint(frames_A_total - n_frames + 1)
            end_A = start_A + n_frames
            train_data_A.append(data_A[:, start_A:end_A])

            data_B = dataset_B[idx_B]
            frames_B_total = data_B.shape[1]
            assert frames_B_total >= n_frames
            start_B = np.random.randint(frames_B_total - n_frames + 1)
            end_B = start_B + n_frames
            train_data_B.append(data_B[:, start_B:end_B])

        train_data_A = np.array(train_data_A)
        train_data_B = np.array(train_data_B)

        return train_data_A[index], train_data_B[index]

    def __len__(self):
        return min(len(self.datasetA), len(self.datasetB))


# if __name__ == '__main__':
#     trainA = np.random.randn(162, 24, 554)
#     trainB = np.random.randn(158, 24, 554)
#     dataset = trainingDataset(trainA, trainB)
#     trainLoader = torch.utils.data.DataLoader(dataset=dataset,
#                                               batch_size=2,
#                                               shuffle=True)
#     for epoch in range(10):
#         for i, (trainA, trainB) in enumerate(trainLoader):
#             print(trainA.shape, trainB.shape)
