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
import copy

# ----------------------------
seed = 42
torch.manual_seed(seed)
random.seed(seed)

def train_scikitlearn_regressor(mlp_reg, X_train_feat, Y_train, X_valid_feat, Y_valid, X_test_feat, Y_test, epochs=15):
    batch_size, train_loss, valid_loss, test_loss, mean_loss = 5000, [], [], [], []
    early_stopper = Sklearn_EarlyStopping(patience=10)
    for _ in tqdm(range(epochs)):
        for b in range(0, len(Y_train), batch_size):
            X_batch, y_batch = X_train_feat[b:b + batch_size], Y_train[b:b + batch_size]
            batch_mean = sum(y_batch) / len(y_batch)
            mlp_reg.partial_fit(X_batch, y_batch)
            train_loss.append(mlp_reg.loss_)
            valid_loss.append(mean_squared_error(Y_valid, mlp_reg.predict(X_valid_feat)))
            test_loss.append(mean_squared_error(Y_test, mlp_reg.predict(X_test_feat)))
            mean_loss.append(mean_squared_error(y_batch, [batch_mean for elem in y_batch]))
        if early_stopper.step(valid_loss[-1], mlp_reg):
            print("Early stopping triggered.")
            mlp_reg = early_stopper.restore_best_weights()
            break
    return mlp_reg, train_loss, valid_loss, test_loss, mean_loss

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

    def restore_best_weights(self, model):
        model.load_state_dict(self.best_state)
class Sklearn_EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = copy.deepcopy(model)
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

    def restore_best_weights(self):
        return self.best_state

class MLP3LayersPredictor(nn.Module):
    def __init__(
        self, in_dim: int, hidden_dim_multiplier: float = 0.5, out_dim: int = 1
    ):
        super().__init__()
        hidden_dim = int(hidden_dim_multiplier * in_dim)
        self.lin_node = nn.Linear(in_dim, hidden_dim)
        self.lin_node2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.lin_node3 = nn.Linear(hidden_dim//2, hidden_dim//8)
        self.out = nn.Linear(hidden_dim//8, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin_node(x)
        x = x.relu()
        x = self.lin_node2(x)
        x = x.relu()
        x = self.lin_node3(x)
        x = x.relu()
        x = self.out(x)
        x = x.sigmoid()
        return x
    def predict(self, x):
        return self.forward(torch.tensor(x).float()).detach().squeeze(1).numpy()

class MLPRegressor(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes):
        super(MLPRegressor, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_layer_sizes[0]))
        layers.append(nn.ReLU()) 
        # Hidden layers
        for i in range(len(hidden_layer_sizes) - 1):
            layers.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_layer_sizes[-1], 1))   
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        return self.forward(torch.tensor(x).float()).detach().squeeze(1).numpy()
def train_mlp(model, X_train_feat, Y_train, X_valid_feat, Y_valid, X_test_feat, Y_test, lr=1e-4, epochs=15):
    batch_size, train_loss, valid_loss, test_loss, mean_loss = 5000, [], [], [], []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion_reg = nn.MSELoss()
    early_stopper = EarlyStopping(patience=10)
    for epoch in tqdm(range(epochs)):
        for b in range(0, len(Y_train), batch_size):
            X_batch, y_reg = X_train_feat[b:b + batch_size], Y_train[b:b + batch_size]
            batch_mean = sum(y_reg) / len(y_reg)
            reg_out= model(torch.tensor(X_batch).float())
            # Loss calculations
            loss_reg = criterion_reg(reg_out.squeeze(1), torch.tensor(y_reg))
            # Weighted multi-task loss
            train_loss_val = loss_reg 
            train_loss.append(train_loss_val.detach().numpy())
            val_reg_out = model(torch.tensor(X_valid_feat).float())
            valid_loss.append(criterion_reg(val_reg_out.detach().squeeze(1), torch.tensor(Y_valid)) )
            test_reg_out = model(torch.tensor(X_test_feat).float())
            test_loss.append(criterion_reg(test_reg_out.detach().squeeze(1), torch.tensor(Y_test)))
            mean_loss.append(mean_squared_error(y_reg, [batch_mean for elem in y_reg]))
            if epochs>(epoch+1):
                train_loss_val.backward()
                optimizer.step()
        if early_stopper.step(valid_loss[-1], model):
            print("Early stopping triggered.")
            early_stopper.restore_best_weights(model)
            break
    return model, train_loss, valid_loss, test_loss, mean_loss


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

