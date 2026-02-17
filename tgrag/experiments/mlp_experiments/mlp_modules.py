# ----------------------------
# Required imports
# ----------------------------
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tqdm import tqdm
import random
import copy
from sklearn.metrics import log_loss
import torch.nn.functional as F
from torch import Tensor, nn
from samplers import BalancedBatchSampler, BinaryClassificationDataset,BalancedRegClsBatchSampler,MultiTaskDataset
# ----------------------------
seed = 42
torch.manual_seed(seed)
random.seed(seed)

def train_scikitlearn_regressor(mlp_reg, X_train_feat, Y_train, X_valid_feat, Y_valid, X_test_feat, Y_test, epochs=15):
    batch_size, train_loss, valid_loss, test_loss, mean_loss = 5000, [], [], [], []
    early_stopper = Sklearn_EarlyStopping(patience=5)
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
            logging.info("Early stopping triggered.")
            mlp_reg = early_stopper.restore_best_weights()
            break
    return mlp_reg, train_loss, valid_loss, test_loss, mean_loss
def train_scikitlearn_classifier(mlp_clf, X_train_feat, Y_train, X_valid_feat, Y_valid, X_test_feat, Y_test, epochs=15):
    batch_size, train_loss, valid_loss, test_loss = 5000, [], [], []
    early_stopper = Sklearn_EarlyStopping(patience=5)
    all_classes = np.unique(Y_train)
    for i in tqdm(range(epochs)):
        for b in range(0, len(Y_train), batch_size):
            X_batch, y_batch = X_train_feat[b:b + batch_size], Y_train[b:b + batch_size]
            if i == 0:
                mlp_clf.partial_fit(X_batch, y_batch, classes=all_classes)
            else:
                mlp_clf.partial_fit(X_batch, y_batch)
            train_loss.append(mlp_clf.loss_)
            valid_loss.append(log_loss(Y_valid, mlp_clf.predict_proba(X_valid_feat)))
            test_loss.append(log_loss(Y_test, mlp_clf.predict_proba(X_test_feat)))
        if early_stopper.step(valid_loss[-1], mlp_clf):
            logging.info("Early stopping triggered.")
            mlp_clf = early_stopper.restore_best_weights()
            break
    return mlp_clf, train_loss, valid_loss, test_loss,None
def train_scikitlearn_classifier_unbalanced(mlp_clf, X_train_feat, Y_train, X_valid_feat, Y_valid, X_test_feat, Y_test, epochs=15):
    batch_size, train_loss, valid_loss, test_loss = 5000, [], [], []
    sampler = BalancedBatchSampler(Y_train, batch_size=batch_size)
    dataset = BinaryClassificationDataset(X_train_feat, Y_train)    
    early_stopper = Sklearn_EarlyStopping(patience=5)
    all_classes = np.unique(Y_train)
    for i in tqdm(range(epochs)):
        train_loader = DataLoader(dataset, batch_sampler=sampler)
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            if i == 0:
                mlp_clf.partial_fit(X_batch, y_batch, classes=all_classes)
            else:
                mlp_clf.partial_fit(X_batch, y_batch)
            train_loss.append(mlp_clf.loss_)
            valid_loss.append(log_loss(Y_valid, mlp_clf.predict_proba(X_valid_feat)))
            test_loss.append(log_loss(Y_test, mlp_clf.predict_proba(X_test_feat)))
        if early_stopper.step(valid_loss[-1], mlp_clf):
            logging.info("Early stopping triggered.")
            mlp_clf = early_stopper.restore_best_weights()
            break
    return mlp_clf, train_loss, valid_loss, test_loss,None

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
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
    def __init__(self, patience=5, min_delta=0.0):
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

        self.activation=nn.ReLU()
        # self.activation=nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:        
        x = self.lin_node(x)        
        x = self.activation(x)
        x = self.lin_node2(x)
        x = self.activation(x)
        x = self.lin_node3(x)
        x = self.activation(x)
        x = self.out(x)
        x = x.sigmoid()
        return x
    def predict(self, x):
        return self.forward(torch.tensor(x).float()).detach().squeeze(1).numpy()
class LabelPredictor(nn.Module):
    def __init__(
        self, in_dim: int, hidden_dim_multiplier: float = 0.5, out_dim: int = 2
    ):
        super().__init__()
        hidden_dim = int(hidden_dim_multiplier * in_dim)
        # hidden_dim=64
        self.lin_node = nn.Linear(in_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)
        self.activation=nn.ReLU()
        # self.activation=nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.lin_node(x)
        x = self.activation(x)
        x = self.out(x)
        return torch.log_softmax(x, dim=-1)
    def predict(self, x):
        return self.forward(torch.tensor(x).float()).argmax(dim=-1) 
class MLPRegressor(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes):
        super(MLPRegressor, self).__init__()
        self.activation=nn.ReLU()
        # self.activation=nn.GELU()
        layers = []
        layers.append(nn.Linear(input_size, hidden_layer_sizes[0]))
        layers.append(self.activation) 
        # Hidden layers
        for i in range(len(hidden_layer_sizes) - 1):
            layers.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i+1]))
            layers.append(self.activation)
        layers.append(nn.Linear(hidden_layer_sizes[-1], 1))   
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        return self.forward(torch.tensor(x).float()).detach().squeeze(1).numpy()
def train_mlp(model, X_train_feat, Y_train, X_valid_feat, Y_valid, X_test_feat, Y_test, lr=1e-4, epochs=15):
    model.train()
    batch_size, train_loss, valid_loss, test_loss, mean_loss = len(Y_train)//5, [], [], [], []
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
            logging.info("Early stopping triggered.")
            early_stopper.restore_best_weights(model)
            break
    return model, train_loss, valid_loss, test_loss, mean_loss

def train_classifier(model, X_train_feat, Y_train, X_valid_feat, Y_valid, X_test_feat, Y_test, lr=1e-4, epochs=15):
    model.train()
    batch_size, train_loss_lst, valid_loss_lst, test_loss_lst = len(Y_train)//5, [], [], []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion_clf =F.nll_loss
    early_stopper = EarlyStopping(patience=5)
    for epoch in tqdm(range(epochs)):
        for b in range(0, len(Y_train), batch_size):
            optimizer.zero_grad()
            X_batch, y_reg = X_train_feat[b:b + batch_size], Y_train[b:b + batch_size]
            batch_mean = sum(y_reg) / len(y_reg)
            clf_out= model(torch.tensor(X_batch).float())                     
            train_loss= criterion_clf(clf_out.float(), torch.tensor(y_reg))               
            if epochs>(epoch+1):
                train_loss.backward()
                optimizer.step()

        with torch.no_grad():
            train_loss_lst.append(train_loss.detach().numpy())
            val_clf_out = model(torch.tensor(X_valid_feat).float())
            # val_clf_out=val_clf_out.argmax(dim=-1)            
            valid_loss_lst.append(criterion_clf(val_clf_out.detach().float().squeeze(), torch.tensor(Y_valid).squeeze()).detach().numpy())
            test_clf_out = model(torch.tensor(X_test_feat).float())
            # test_clf_out=test_clf_out.argmax(dim=-1)
            test_loss_lst.append(criterion_clf(test_clf_out.detach().float().squeeze(), torch.tensor(Y_test).squeeze()).detach().numpy())
            logging.info(f"Epoch={epoch}\t Batch={b}\t train_loss={train_loss_lst[-1]}\t valid_loss={valid_loss_lst[-1]}\t test_loss={test_loss_lst[-1]}")

        if early_stopper.step(valid_loss_lst[-1], model):
            logging.info("Early stopping triggered.")
            early_stopper.restore_best_weights(model)
            break
    return model, train_loss_lst, valid_loss_lst, test_loss_lst, None
    
def train_classifier_unbalanced(model, X_train_feat, Y_train, X_valid_feat, Y_valid, X_test_feat, Y_test, lr=1e-4, epochs=15,batch_size=int(1e4)):
    sampler = BalancedBatchSampler(Y_train, batch_size=batch_size)
    dataset = BinaryClassificationDataset(X_train_feat, Y_train)  
    # for batch_idx, (X_batch, y_reg) in enumerate(train_loader):
    #     logging.info(f"Batch {batch_idx}: X_batch shape: {X_batch.shape}, y_reg shape: {y_reg.shape}")
    model.train()
    batch_size, train_loss_lst, valid_loss_lst, test_loss_lst = len(Y_train)//5, [], [], []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion_clf =F.nll_loss
    early_stopper = EarlyStopping(patience=5)
    for epoch in tqdm(range(epochs)):
        train_loader = DataLoader(dataset, batch_sampler=sampler)
        for batch_idx, (X_batch, y_reg) in enumerate(train_loader):
            optimizer.zero_grad()
            clf_out= model(torch.tensor(X_batch).float())                     
            train_loss= criterion_clf(clf_out.float(), torch.tensor(y_reg).long())               
            if epochs>(epoch+1):
                train_loss.backward()
                optimizer.step()

        with torch.no_grad():
            train_loss_lst.append(train_loss.detach().numpy())
            val_clf_out = model(torch.tensor(X_valid_feat).float())
            # val_clf_out=val_clf_out.argmax(dim=-1)            
            valid_loss_lst.append(criterion_clf(val_clf_out.detach().float().squeeze(), torch.tensor(Y_valid).squeeze()).detach().numpy())
            test_clf_out = model(torch.tensor(X_test_feat).float())
            # test_clf_out=test_clf_out.argmax(dim=-1)
            test_loss_lst.append(criterion_clf(test_clf_out.detach().float().squeeze(), torch.tensor(Y_test).squeeze()).detach().numpy())
            logging.info(f"Epoch={epoch}\t Batch={batch_idx}\t train_loss={train_loss_lst[-1]}\t valid_loss={valid_loss_lst[-1]}\t test_loss={test_loss_lst[-1]}")

        if early_stopper.step(valid_loss_lst[-1], model):
            logging.info("Early stopping triggered.")
            early_stopper.restore_best_weights(model)
            break
    return model, train_loss_lst, valid_loss_lst, test_loss_lst, None


class MultiTaskMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=[128, 64]):
        super(MultiTaskMLP, self).__init__()
        self.activation=nn.ReLU()
        # self.activation=nn.GELU()
        # Shared feature extractor
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(self.activation)
        for idx in range(1,len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[idx-1], hidden_dims[idx]))
            layers.append(self.activation)
        self.trunk = nn.Sequential(*layers)

        # Task-specific heads
        self.reg_pre_head = nn.Linear(hidden_dims[-1], hidden_dims[-1]//2)
        self.reg_head = nn.Linear(hidden_dims[-1]//2, 1)  # Regression
        
        self.cls_pre_head = nn.Linear(hidden_dims[-1], hidden_dims[-1]//2)  
        self.cls_head = nn.Linear(hidden_dims[-1]//2, num_classes)  # Multi-class classification
    def forward(self, x,y):
        z_x = self.trunk(x)
        z_y = self.trunk(y)

        reg_out = self.reg_pre_head(z_x)  # (batch, 1)
        reg_out = self.reg_head(reg_out)  # (batch, 1)
        reg_out=reg_out.sigmoid()

        cls_logits = self.cls_pre_head(z_y)  # (batch, num_classes)
        cls_logits = self.cls_head(cls_logits)
        cls_pred=torch.log_softmax(cls_logits, dim=-1)

        return reg_out, cls_logits, cls_pred

    def predict(self, x,y):
        reg_out, cls_logits, cls_pred = self.forward(torch.tensor(x).float(),torch.tensor(y).float())
        cls_pred=cls_pred.argmax(dim=-1) 
        return reg_out.detach().squeeze(1).numpy(),cls_pred


class MLPHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=[128], dropout=0.2):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(d, h))
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class MultiTaskMLP(nn.Module):
    def __init__(self, input_dim, num_classes,hidden_dims=[32]):
        super().__init__()

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            # nn.Linear(128, 64),
            # nn.ReLU(),
        )

        # Task-specific deep heads
        self.reg_head = MLPHead(128, 1)
        self.cls_head = MLPHead(128, num_classes)

    def forward(self, x, task=None):
        z = self.trunk(x)
        if task == "reg":
            reg_out= self.reg_head(z)
            reg_out=reg_out.sigmoid()   
            return reg_out
        elif task == "cls":
            clf_out=self.cls_head(z)
            return torch.log_softmax(clf_out, dim=-1)
        else:
            # default: return both
            return self.reg_head(z).sigmoid(), torch.log_softmax(self.cls_head(z),dim=-1)
    def predict(self, x,task=None):        
        out= self.forward(torch.tensor(x).float(),task)
        if task == "cls":
            out=out.argmax(dim=-1) 
        return out

class URLLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # two learnable parameters
        self.alpha = nn.Parameter(torch.zeros(2))  # [cls, reg]
    def forward(self, loss_cls, loss_reg):
        # softmax to enforce constraint
        weights = torch.softmax(self.alpha, dim=0)
        w_cls = weights[0]
        w_reg = weights[1]
        total_loss = loss_cls / w_cls + loss_reg / w_reg
        return total_loss


def train_multihead_unbalanced(model, reg_X_train_feat, reg_Y_train, reg_X_valid_feat, reg_Y_valid, reg_X_test_feat, reg_Y_test,
                                cls_X_train_feat, cls_Y_train, cls_X_valid_feat, cls_Y_valid, cls_X_test_feat, cls_Y_test, lr=1e-4, epochs=15,batch_size=int(1e4),use_URL_Loss=False):
    sampler = BalancedRegClsBatchSampler(reg_Y_train,cls_Y_train, batch_size=batch_size)
    dataset = MultiTaskDataset(reg_X_train_feat,cls_X_train_feat, reg_Y_train,cls_Y_train)  
    model.train()
    batch_size, train_loss_lst, valid_loss_lst, test_loss_lst,mean_loss_lst = len(cls_Y_train)//5, [], [], [],[]
    train_loss_clf_lst,valid_loss_clf_lst, test_loss_clf_lst=[],[],[]

    if use_URL_Loss:
        url_loss = URLLoss()
        optimizer=torch.optim.Adam([{"params": model.parameters()},{"params": url_loss.parameters(), "lr": 1e-1}],lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion_clf =F.nll_loss
    criterion_reg = nn.MSELoss()
    early_stopper = EarlyStopping(patience=5)
    
    for epoch in tqdm(range(epochs)):
        train_loader = DataLoader(dataset, batch_sampler=sampler)
        for batch_idx, (reg_X_batch,cls_X_batch,reg_y,cls_y) in enumerate(train_loader):
            optimizer.zero_grad()
            reg_out= model(torch.tensor(reg_X_batch).float(),task="reg")                     
            reg_loss= criterion_reg(reg_out.squeeze(1), torch.tensor(reg_y))   
            reg_loss_val=reg_loss.detach().numpy()
            # if epochs>(epoch+1):
            #     train_loss.backward()
            #     optimizer.step()
            # optimizer.zero_grad()

            clf_out= model(torch.tensor(cls_X_batch).float(),task="cls")                     
            clf_loss= criterion_clf(clf_out.float(), torch.tensor(cls_y).long()) 
            if use_URL_Loss:
                loss = url_loss(reg_loss, clf_loss)                     
            else:
                loss = 0.2 * reg_loss + 0.8 * clf_loss   
            if epochs>(epoch+1):
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            clf_loss_val=reg_loss.detach().numpy()   
            train_loss_lst.append(reg_loss_val) 
            train_loss_clf_lst.append(clf_loss_val)
            if use_URL_Loss: 
                logging.info(f"URL Loss Learned Weights(cls,reg)={torch.softmax(url_loss.alpha, dim=0)}")    

            val_cls_out = model(torch.tensor(cls_X_valid_feat).float(),task="cls")
            valid_loss_clf_lst.append(criterion_clf(val_cls_out.detach().float().squeeze(), torch.tensor(cls_Y_valid).squeeze()).detach().numpy())

            val_reg_out = model(torch.tensor(reg_X_valid_feat).float(),task="reg")            
            valid_loss_lst.append(criterion_reg(val_reg_out.detach().float().squeeze(1), torch.tensor(reg_Y_valid)))

            test_cls_out = model(torch.tensor(cls_X_test_feat).float(),task="cls")
            test_loss_clf_lst.append(criterion_clf(test_cls_out.detach().float().squeeze(), torch.tensor(cls_Y_test).squeeze()).detach().numpy())

            test_reg_out = model(torch.tensor(reg_X_test_feat).float(),task="reg")
            test_loss_lst.append(criterion_reg(test_reg_out.detach().float().squeeze(1), torch.tensor(reg_Y_test)))                        

            logging.info(f"Epoch={epoch}\t  train_loss={train_loss_lst[-1]}||\t val_loss_reg={valid_loss_lst[-1]}\t val_loss_clf={valid_loss_clf_lst[-1]}\t|| test_loss={test_loss_lst[-1]}\t test_loss={test_loss_clf_lst[-1]}")                     

        if early_stopper.step(valid_loss_lst[-1]*.5+valid_loss_clf_lst[-1]*.5, model):
            logging.info("Early stopping triggered.")
            early_stopper.restore_best_weights(model)
            break
    return model,train_loss_lst,train_loss_clf_lst,valid_loss_lst,valid_loss_clf_lst, test_loss_lst,test_loss_clf_lst, mean_loss_lst 

def train_multihead_v2(model, X_train_feat_reg, Y_train_reg, X_valid_feat_reg, Y_valid_reg, X_test_feat_reg, Y_test_reg,
                           X_train_feat_clf, Y_train_clf, X_valid_feat_clf, Y_valid_clf, X_test_feat_clf, Y_test_clf, lr=1e-4, epochs=15,
                    num_classes=3, w_reg=0.5, w_cls=0.5):
    batch_size, train_loss_lst, valid_loss_lst, test_loss_lst, mean_loss_lst = 5000, [], [], [], []
    train_loss_clf_lst,valid_loss_clf_lst, test_loss_clf_lst=[],[],[]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion_reg = nn.MSELoss()
    criterion_clf = F.nll_loss
    w_reg = w_reg
    w_cls = w_cls
    sampler = BalancedBatchSampler(Y_train_clf, batch_size=batch_size)
    dataset = BinaryClassificationDataset(X_train_feat_clf, Y_train_clf)      
    model.train()  
    early_stopper = EarlyStopping(patience=5)   
    for epoch in tqdm(range(epochs)):
        for b in range(0, len(Y_train_reg), batch_size):
            X_batch_reg, y_reg = X_train_feat_reg[b:b + batch_size], Y_train_reg[b:b + batch_size]
            train_loader_clf = DataLoader(dataset, batch_sampler=sampler)
            batch_mean = sum(y_reg) / len(y_reg)
            for batch_idx_clf, (X_batch_clf, y_clf) in enumerate(train_loader_clf):
                optimizer.zero_grad()
                reg_out, cls_logits, clf_out = model(torch.tensor(X_batch_reg).detach().float(),torch.tensor(X_batch_clf).detach().float())              
                loss_reg = criterion_reg(reg_out.squeeze(1), torch.tensor(y_reg))*0.5
                loss_clf= criterion_clf(clf_out.float(), torch.tensor(y_clf).long())*0.5
                train_loss_val = w_reg * loss_reg + w_cls * loss_clf     
                if epochs>(epoch+1):
                    loss_reg.backward()
                    loss_clf.backward()
                    optimizer.step()
        
        with torch.no_grad():
            train_loss_lst.append(loss_reg.detach().numpy())
            train_loss_clf_lst.append(loss_clf.detach().numpy())            
            val_reg_out, val_cls_logits, val_clf_out=model(torch.tensor(X_valid_feat_reg).float(),torch.tensor(X_valid_feat_clf).float())              
            val_loss_reg = criterion_reg(val_reg_out.squeeze(1), torch.tensor(Y_valid_reg))
            val_loss_clf= criterion_clf(val_clf_out.float(), torch.tensor(Y_valid_clf).long())
            valid_loss_lst.append(val_loss_reg)
            valid_loss_clf_lst.append(val_loss_clf)

            test_reg_out, test_cls_logits, test_clf_out=model(torch.tensor(X_test_feat_reg).float(),torch.tensor(X_test_feat_clf).float())              
            test_loss_reg = criterion_reg(test_reg_out.squeeze(1), torch.tensor(Y_test_reg))
            test_loss_clf= criterion_clf(test_clf_out.float(), torch.tensor(Y_test_clf).long())
            test_loss_lst.append(test_loss_reg)
            test_loss_clf_lst.append(test_loss_clf)
            
            logging.info(f"Epoch={epoch}\t  train_loss={train_loss_lst[-1]}||\t val_loss_reg={valid_loss_lst[-1]}\t val_loss_clf={valid_loss_clf_lst[-1]}\t|| test_loss={test_loss_lst[-1]}\t test_loss={test_loss_clf_lst[-1]}")                     
            mean_loss_lst.append(mean_squared_error(y_reg, [batch_mean for elem in y_reg]))            
        if early_stopper.step(valid_loss_lst[-1]*0.5+valid_loss_clf_lst[-1]*0.5, model):
            logging.info("Early stopping triggered.")
            early_stopper.restore_best_weights(model)
            break
    return model,train_loss_lst,train_loss_clf_lst,valid_loss_lst,valid_loss_clf_lst, test_loss_lst,test_loss_clf_lst, mean_loss_lst 


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
        logging.info(f"loss_reg={loss_reg}\tcls_logits={cls_logits}")
        # Weighted multi-task loss
        loss = w_reg * loss_reg + w_cls * loss_cls
        logging.info(loss)
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    model = MultiTaskMLP(input_dim=100, num_classes=2, hidden_dims=[1024, 256, 32])
    train_multihead_v0(model, 100)

