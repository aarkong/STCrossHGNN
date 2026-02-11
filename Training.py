import numpy as np
import pandas as pd
import torch
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

set_seed(42)

import os.path as osp
from sklearn.utils.class_weight import compute_class_weight
import time
from tqdm import tqdm

import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import KFold

from Model import HGNNClassifier
from Dataset import ConnectivityData
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RobustFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(RobustFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        ce_loss = torch.clamp(ce_loss, min=0, max=100)
        
        pt = torch.exp(-ce_loss)
        pt = torch.clamp(pt, min=1e-7, max=1-1e-7)

        alpha_t = self.alpha[targets.long()]
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        if torch.isnan(focal_loss).any():
            return torch.tensor(1.0, device=inputs.device, requires_grad=True)
            
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

TRAINING_STRATEGY = 'fixed'  # 'fixed' or 'early_stopping'
FIXED_NUM_EPOCHS = 100
EARLY_STOPPING_MAX_EPOCHS = 120
EARLY_STOPPING_PATIENCE = 50
EARLY_STOPPING_MIN_DELTA = 0.01
EARLY_STOPPING_LOSS_THRESHOLD = 0.8

def HGNN_train(train_loader, fold, epoch, class_weights):
    model.train()
    loss_all = 0
    scaler = GradScaler()
    
    accumulation_steps = 4
    
    for batch_idx, data in enumerate(tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch} Batch",
                     position=2, leave=False)):
        data = data.to(device)
        
        with autocast():
            output = model(data)
            logits = output[0] if isinstance(output, tuple) else output
            loss = RobustFocalLoss(alpha=class_weights, gamma=2.0)(logits, data.y)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf loss detected, skipping batch")
            continue

        scaled_loss = loss / accumulation_steps
        scaler.scale(scaled_loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        loss_all += data.num_graphs * loss.item()

    if (batch_idx + 1) % accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    return loss_all / len(train_loader.dataset)


def HGNN_test(loader, fold, epoch, class_weights):
    model.eval()
    pred_probs = []
    label = []
    loss_all = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)

            loss = RobustFocalLoss(alpha=class_weights, gamma=2.0)(output, data.y)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf validation loss detected")
                loss = torch.tensor(1.0, device=device)
                
            loss_all += data.num_graphs * loss.item()
            probs = F.softmax(output, dim=1)[:, 1]
            pred_probs.append(probs)
            label.append(data.y)
    
    best_threshold = 0.5
    if epoch > 0:
        from sklearn.metrics import roc_curve
        y_true = torch.cat(label).cpu().numpy()
        y_score = torch.cat(pred_probs).cpu().numpy()
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        youden_index = tpr - fpr
        best_threshold = thresholds[np.argmax(youden_index)]
    
    y_pred = (torch.cat(pred_probs).cpu().numpy() >= best_threshold).astype(int)
    y_true = torch.cat(label).cpu().numpy()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return sensitivity, precision, specificity, accuracy, loss_all / len(loader.dataset)

import os
cache_path = ''
if os.path.exists(cache_path):
    os.remove(cache_path)
    print(f"已删除缓存文件: {cache_path}")
save_dir = ''
os.makedirs(save_dir, exist_ok=True)
dataset = ConnectivityData('')

model = HGNNClassifier(
    num_features=dataset.num_features,  
    num_classes=2,
    dropout=0.7,
    lstm_hidden=256,
    time_steps=4,
    hgnn_hid_dim=256,
    mask_rate=0.2,
    aug_ratio=0.05,
    seed=42
)
model = model.to(device)
labels = dataset.get_labels()
assert dataset.num_features == 90, f"特征维度不匹配: {dataset.num_features} != 90"
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)  # 固定随机种子
eval_metrics = []

for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
    test_dataset = dataset[test_idx.tolist()]
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    train_val_dataset = dataset[train_val_idx.tolist()]
    train_idx, val_idx = train_test_split(
        np.arange(len(train_val_dataset)),
        test_size=0.3,
        stratify=labels[train_val_idx],
        random_state=42
    )

    train_val_labels = labels[train_val_idx]

model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=0.00001,
    weight_decay=1e-3
)

from torch.optim.lr_scheduler import LinearLR, SequentialLR

warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=10)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=20, T_mult=2, eta_min=1e-6)
scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[10])

train_val_dataset = dataset[train_val_idx.tolist()]
test_dataset = dataset[test_idx.tolist()]
train_idx, val_idx = train_test_split(
    np.arange(len(train_val_dataset)), 
    test_size=0.3, 
    stratify=train_val_labels,
    random_state=42
)
train_dataset = train_val_dataset[train_idx.tolist()]
val_dataset = train_val_dataset[val_idx.tolist()]
val_labels = np.array([data.y.item() for data in val_dataset])
print(f"Fold {fold_idx+1} 验证集类别分布: {np.bincount(val_labels)}")
    
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
num_workers=4, pin_memory=True, worker_init_fn=lambda x: 42)

if TRAINING_STRATEGY == 'fixed':
    epochs = FIXED_NUM_EPOCHS
else:
    epochs = EARLY_STOPPING_MAX_EPOCHS

best_f1 = -np.inf
counter = 0

total_start_time = time.time()

with tqdm(total=10, desc="Cross Validation", position=0, leave=True) as fold_pbar:
    for outer_fold in range(10):
        fold_start_time = time.time()
        
        fold_pbar.set_description(f"Fold {outer_fold+1}/10")

        train_idx, val_idx = list(skf.split(np.zeros(len(labels)), labels))[outer_fold]
        train_dataset = dataset[train_idx]
        val_dataset = dataset[val_idx]

        fold_train_labels = np.array([dataset[i].y.item() for i in train_idx])
        class_counts = np.bincount(fold_train_labels)
        pos_ratio = class_counts[1] / len(fold_train_labels)

        if pos_ratio > 0.6:
            class_weights = [1.5, 1.0]
        elif pos_ratio < 0.4:
            class_weights = [1.0, 1.5]
        else:
            class_weights = [1.2, 1.2]
        
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

        fold_pbar.update(1)
        def worker_init_fn(worker_id):
            seed = 42 + worker_id
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
        num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
        num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

        best_f1 = -np.inf
        counter = 0

        epoch_pbar = tqdm(range(epochs), desc=f"Fold {outer_fold+1} Epoch", position=1, leave=True)

        early_stop_epoch = epochs
        for epoch in epoch_pbar:
            epoch_start_time = time.time()

            train_loss = HGNN_train(train_loader, outer_fold+1, epoch+1, class_weights)
            val_sen, val_pre, val_spe, val_acc, val_loss = HGNN_test(val_loader, outer_fold + 1, epoch + 1, class_weights)
            val_f1 = 2 * (val_pre * val_sen) / (val_pre + val_sen + 1e-6)  # 使用精确率和召回率
            scheduler.step()

            should_stop = False
            
            if TRAINING_STRATEGY == 'early_stopping':
                current_metric = val_f1
                

                if val_loss > EARLY_STOPPING_LOSS_THRESHOLD:
                    should_stop = True
                    print(f"验证损失{val_loss:.3f}超过阈值{EARLY_STOPPING_LOSS_THRESHOLD}，立即停止\n")
                
                elif abs(train_loss - val_loss) > 0.5:
                    should_stop = True
                    print(f"训练-验证损失差距过大({abs(train_loss - val_loss):.3f})，停止\n")

                elif current_metric <= best_f1 + EARLY_STOPPING_MIN_DELTA:
                    counter += 1
                    if counter >= EARLY_STOPPING_PATIENCE:
                        should_stop = True
                        print(f"连续{EARLY_STOPPING_PATIENCE}个epoch无改进，早停\n")
                else:
                    best_f1 = max(best_f1, current_metric)
                    counter = 0
            
            if should_stop:
                epoch_pbar.set_postfix_str(f"{epoch_pbar.postfix} - 早停于epoch {epoch+1}")
                early_stop_epoch = epoch + 1
                break

        epoch_pbar.close()

        fold_end_time = time.time()
        fold_duration = fold_end_time - fold_start_time

        test_sen, test_pre, test_spe, test_acc, _ = HGNN_test(test_loader, outer_fold+1, 0, class_weights)
        
        fold_pbar.set_postfix({'time': f'{fold_duration:.1f}s'})
        
        eval_metrics.append([test_sen, test_spe, test_acc])

        print(f"\n=== Fold {outer_fold+1} 训练结果 ===")
        print(f"测试集指标: 灵敏度={test_sen:.4f}, 特异度={test_spe:.4f}, 准确率={test_acc:.4f}")
        print(f"早停位置: {early_stop_epoch}/{epochs} epochs")
        print(f"================================\n")

fold_pbar.close()

total_end_time = time.time()
total_duration = total_end_time - total_start_time

eval_metrics = np.array(eval_metrics)
eval_df = pd.DataFrame(eval_metrics, columns=['SEN', 'SPE', 'ACC'], 
                       index=[f'Fold_{i+1:02d}' for i in range(10)]).round(4)
print("\n" + "="*50)
print("训练完成汇总:")
print(eval_df)
print(f'Average SEN: {eval_metrics[:,0].mean():.4f} ± {eval_metrics[:,0].std():.4f}')
print(f'Average SPE: {eval_metrics[:,1].mean():.4f} ± {eval_metrics[:,1].std():.4f}')
print(f'Average ACC: {eval_metrics[:,2].mean():.4f} ± {eval_metrics[:,2].std():.4f}')
print(f"\n总训练时间: {total_duration:.2f} 秒 ({total_duration/60:.2f} 分钟)")