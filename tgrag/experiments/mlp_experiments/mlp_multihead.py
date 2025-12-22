# ----------------------------
# Required imports
# ----------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tqdm import tqdm
import random

# ----------------------------
seed = 42
torch.manual_seed(seed)
random.seed(seed)


class MultiTaskMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=[128, 64]):
        super(MultiTaskMLP, self).__init__()

        # Shared feature extractor
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout(0.3))
            # layers.append(nn.BatchNorm1d(h))
            in_dim = h
        self.trunk = nn.Sequential(*layers)

        # Task-specific heads
        self.reg_head = nn.Linear(in_dim, 1)  # Regression
        self.cls_head = nn.Linear(in_dim, num_classes)  # Multi-class classification

    def forward(self, x):
        z = self.trunk(x)
        reg_out = self.reg_head(z)  # (batch, 1)
        cls_logits = self.cls_head(z)  # (batch, num_classes)
        cls_pred = torch.argmax(cls_logits, dim=1)
        return reg_out, cls_logits, cls_pred

    def predict(self, x):
        reg_out, cls_logits, cls_pred = self.forward(torch.tensor(x))
        return reg_out.detach().squeeze(1).numpy()


def train_multihead(model, X_train_feat, Y_train, X_valid_feat, Y_valid, X_test_feat, Y_test, lr=1e-4, epochs=15,
                    num_classes=3, w_reg=0.8, w_cls=0.2):
    batch_size, train_loss, valid_loss, test_loss, mean_loss = 5000, [], [], [], []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion_reg = nn.MSELoss()
    criterion_cls = nn.CrossEntropyLoss()
    w_reg = w_reg
    w_cls = w_cls
    for epoch in tqdm(range(epochs)):
        for b in range(0, len(Y_train), batch_size):
            X_batch, y_reg = X_train_feat[b:b + batch_size], Y_train[b:b + batch_size]
            y_cls = torch.tensor([int(((elem * 10) % 10) / (num_classes + 1)) for elem in y_reg], dtype=torch.long)
            batch_mean = sum(y_reg) / len(y_reg)
            reg_out, cls_logits, cls_pred = model(torch.tensor(X_batch))
            # Loss calculations
            loss_reg = criterion_reg(reg_out.squeeze(1), torch.tensor(y_reg))
            loss_cls = criterion_cls(cls_logits, y_cls)
            # Weighted multi-task loss
            train_loss_val = w_reg * loss_reg + w_cls * loss_cls
            train_loss.append(train_loss_val.detach().numpy())
            val_reg_out, val_cls_logits, val_cls_pred = model(torch.tensor(X_valid_feat))
            valid_loss.append(criterion_reg(val_reg_out.detach().squeeze(1), torch.tensor(Y_valid)) * w_reg)
            test_reg_out, test_cls_logits, test_cls_pred = model(torch.tensor(X_test_feat))
            test_loss.append(criterion_reg(test_reg_out.detach().squeeze(1), torch.tensor(Y_test)) * w_reg)
            mean_loss.append(mean_squared_error(y_reg, [batch_mean for elem in y_reg]))
            if epochs>(epoch+1):
                train_loss_val.backward()
                optimizer.step()
    return model, train_loss, valid_loss, test_loss, mean_loss


def train_multihead_v0(model, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion_reg = nn.MSELoss()
    criterion_cls = nn.CrossEntropyLoss()
    # Task weights
    w_reg = 0.4
    w_cls = 0.6
    # Dummy batch
    x = torch.randn(32, 100)
    y_reg = torch.tensor(np.arange(0, 32), dtype=torch.float64)
    y_cls = torch.tensor([int(elem) % 2 for elem in y_reg], dtype=torch.long)
    y_reg = y_reg.reshape(32, 1).float()
    for i in range(epochs):
        # Forward pass
        reg_out, cls_logits, cls_pred = model(x)
        # Loss calculations
        loss_reg = criterion_reg(reg_out, y_reg)
        loss_cls = criterion_cls(cls_logits, y_cls)
        print(f"loss_reg={loss_reg}\tcls_logits={cls_logits}")
        # Weighted multi-task loss
        loss = w_reg * loss_reg + w_cls * loss_cls
        print(loss)
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    model = MultiTaskMLP(input_dim=100, num_classes=2, hidden_dims=[1024, 256, 32])
    train_multihead_v0(model, 100)

