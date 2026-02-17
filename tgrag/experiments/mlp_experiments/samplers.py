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

class MultiTaskDataset(Dataset):
    def __init__(self, X_reg,X_cls, y_reg, y_cls):
        """
        Parameters
        ----------
        X_reg : torch.Tensor (N, D)
        X_cls : torch.Tensor (N, D)
        y_reg : torch.Tensor (N, 1)
        y_cls : torch.Tensor (N,C)
        """
        self.X_reg = torch.as_tensor(X_reg, dtype=torch.float32)
        self.X_cls = torch.as_tensor(X_cls, dtype=torch.float32)
        self.y_reg = torch.as_tensor(y_reg, dtype=torch.float32)
        self.y_cls =torch.as_tensor(y_cls, dtype=torch.float32)

    def __len__(self):
        return self.X_cls.shape[0]

    def __getitem__(self, idx):
        return self.X_reg[idx[0]],self.X_cls[idx[1]], self.y_reg[idx[0]], self.y_cls[idx[1]]

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

class BalancedRegClsBatchSampler(Sampler):
    def __init__(self, reg_labels,bcls_labels, batch_size):
        self.bcls_labels = np.array(bcls_labels)
        self.reg_labels = np.array(reg_labels)
        self.batch_size = batch_size
        self.half = batch_size // 2
        self.cls_idx_0 = np.where(self.bcls_labels == 0)[0]
        self.cls_idx_1 = np.where(self.bcls_labels == 1)[0]
        self.reg_idx = np.where(self.reg_labels >=0 )[0]
        assert len(self.cls_idx_0) > 0 and len(self.cls_idx_1) > 0 and len(self.reg_idx) > 0

    def __iter__(self):
        np.random.shuffle(self.cls_idx_0)
        np.random.shuffle(self.cls_idx_0)
        np.random.shuffle(self.reg_idx)
        for ptr0 in range(0, len(self.cls_idx_0),self.half):
            bs=self.half
            reg_ptr=ptr0
            rg_bs=self.half
            if self.half + ptr0 > len(self.cls_idx_0):
                bs = len(self.cls_idx_0) - ptr0
                rg_bs=bs
            if rg_bs + reg_ptr > len(self.reg_idx):
                reg_ptr= len(self.reg_idx)-rg_bs

            for ptr1 in range(0, len(self.cls_idx_1),bs):
                batch = np.concatenate([
                    list(zip(self.reg_idx[reg_ptr:reg_ptr + rg_bs],self.cls_idx_0[ptr0:ptr0 + bs])),
                    list(zip(self.reg_idx[reg_ptr:reg_ptr + rg_bs],self.cls_idx_1[ptr1:ptr1 + bs]))])
                np.random.shuffle(batch)
                yield batch.tolist()                
                ptr1 += self.half
            ptr0 += self.half

    def __len__(self):
        return len(self.cls_idx_0) // self.half
