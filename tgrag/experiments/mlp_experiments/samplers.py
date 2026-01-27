import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np

class BinaryClassificationDataset(Dataset):
    def __init__(self, X, y):
        """
        X: Tensor or ndarray of shape [N, D]
        y: Tensor or ndarray of shape [N]
        """
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.half = batch_size // 2

        self.idx_0 = np.where(self.labels == 0)[0]
        self.idx_1 = np.where(self.labels == 1)[0]

        assert len(self.idx_0) > 0 and len(self.idx_1) > 0

    def __iter__(self):
        np.random.shuffle(self.idx_0)
        np.random.shuffle(self.idx_1)
        for ptr0 in range(0, len(self.idx_0),self.half):
            bs=self.half
            if self.half + ptr0 > len(self.idx_0):
                bs = len(self.idx_0) - ptr0
            for ptr1 in range(0, len(self.idx_1),bs):
                batch = np.concatenate([
                    self.idx_0[ptr0:ptr0 + bs ],
                    self.idx_1[ptr1:ptr1 + bs]
                ])
                np.random.shuffle(batch)
                yield batch.tolist()                
                ptr1 += self.half
            ptr0 += self.half

    def __len__(self):
        return len(self.idx_0) // self.half
