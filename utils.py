import os
os.environ["MPLCONFIGDIR"] = os.path.expanduser("/tmp/matplotlib")
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import random
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, cohen_kappa_score
import seaborn as sns
############ Import Network ############
from MIRACLE import MIRACLE
################################################

available_network = [
    'MIRACLE'
]

network_factory_methods = {
    'MIRACLE': MIRACLE,
}

available_paradigm = [
    'Cross'
]

# ------------------------------------------ NORMALIZATIONs ------------------------------------------ #
def _z_score_normalization(data):
    mean = data.mean(dim=None, keepdim=True)
    std = data.std(dim=None, keepdim=True)
    return mean, std

def load_normalizations(saved_path):
    mean, std = None, None
    if os.path.exists(f'{saved_path}_mean_data.pt'):
        mean = torch.load(f'{saved_path}_mean_data.pt')
    if os.path.exists(f'{saved_path}_std_data.pt'):
        std = torch.load(f'{saved_path}_std_data.pt')
    return mean, std

def normalization_z_score_unique(data):
    mean, std = _z_score_normalization(data)
    return mean, std, None, None

normalization_factory_methods = {
    'Z_Score_unique': normalization_z_score_unique,
}

available_normalization = [
    'Z_Score_unique'
]

def data_labels_from_subset(subset):
    data = torch.stack([subset[i][0] for i in range(len(subset))])
    labels = torch.stack([subset[i][1] for i in range(len(subset))])
    labels_subjects = torch.stack([subset[i][2] for i in range(len(subset))])
    return data, labels, labels_subjects
        
def normalize_subset(train_subset, val_subset, normalization_function):
    train_data, train_labels, train_labels_subjects = data_labels_from_subset(train_subset)
    val_data, val_labels, val_labels_subjects = data_labels_from_subset(val_subset)

    mean, std, min_, max_ = normalization_function(train_data)
    
    if mean != None:
        train_data = (train_data - mean) / std
        val_data = (val_data - mean) / std
    else:
        train_data = (train_data - min_)/(max_-min_)
        val_data = (val_data - min_)/(max_-min_)

    train_tensor = TensorDataset(train_data, train_labels, train_labels_subjects)
    val_tensor = TensorDataset(val_data, val_labels, val_labels_subjects)

    return train_tensor, val_tensor

# ------------------------------------------ CREATE TENSORS ------------------------------------------ #
 
def create_tensors_subjects(dataset_path):
    dataset = np.load(dataset_path, allow_pickle=True)
    data_tensor = []
    data_list = dataset['data']
    for data in data_list:
        data_tensor.append(torch.tensor(data).float().unsqueeze(1))
    labels_tensor = []
    labels_subjects = []
    labels_list = dataset['labels']
    for subject, labels in enumerate(labels_list):
        labels_tensor.append(torch.tensor(labels))
        labels_subjects.append(torch.full_like(labels_tensor[subject], subject))
    
    return data_tensor, labels_tensor, labels_subjects

# ------------------------------------------ FIX SEED ------------------------------------------ #

def fix_seeds(seed=42):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------------------------------ PLOTs ------------------------------------------ #

def plot_losses(list_loss_train, list_loss_train_tasks, list_loss_train_subjects, 
                list_loss_validation, list_loss_validation_tasks, list_loss_validation_subjects, fold=0, path='Results/', reconstruction=False):
    
    min_index = list_loss_validation.index(min(list_loss_validation))
    min_value = list_loss_validation[min_index]
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(range(len(list_loss_train)), list_loss_train, marker='o', label='Train Loss')
    plt.plot(range(len(list_loss_train_tasks)), list_loss_train_tasks, marker='o', label='Train Loss Tasks')
    plt.plot(range(len(list_loss_train_subjects)), list_loss_train_subjects, marker='o', label='Train Loss Reconstruction' if reconstruction else 'Train Loss Subjects')
    
    plt.plot(range(len(list_loss_validation)), list_loss_validation, marker='o', label='Validation Loss')
    plt.plot(range(len(list_loss_validation_tasks)), list_loss_validation_tasks, marker='o', label='Validation Loss Tasks')
    plt.plot(range(len(list_loss_validation_subjects)), list_loss_validation_subjects, marker='o', label='Validation Loss Reconstruction' if reconstruction else 'Validation Loss Subjects')
    
    plt.plot(min_index, min_value, 'gs', label='Minimum Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.title(f'Train and Validation Loss per Epoch in fold {fold}')
    plt.legend()
    plt.savefig(f"./{path}_plot_losses_fold{fold}")
    plt.close()

def plot_confusion_matrix(confusion_matrix, class_names, fold, name, accuracy):
    plt.figure(figsize=(10, 10))
    confusion_matrix = confusion_matrix/confusion_matrix.sum(axis=1, keepdims=True)
    sns.heatmap(confusion_matrix, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 42}, cbar=False)
    plt.ylabel('True label', fontsize=32, fontweight='bold')
    plt.xlabel('Predicted label', fontsize=32, fontweight='bold')
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    # plt.title(f'Confusion Matrix fold: {fold} - Accuracy: {accuracy:.2f}')
    plt.title(f'Balanced Accuracy: {accuracy:.2f}', fontsize=32)
    plt.savefig(f'./{name}_confusion_matrix_{fold}.png')
    plt.close()

def plot_training_complete(fold_performance, name, folds):
    print('Training complete')
    for fold, performance in enumerate(fold_performance):
        with open(f'{name}_validation_log.txt', 'a') as f:
            f.write(f'Fold {fold + 1} - Loss: {performance[0]:.3f}, Loss Tasks: {performance[1]:.3f}, Loss Subjects: {performance[2]:.3f}, '
                    f'F1 Tasks: {performance[3]}, F1 Subjects: {performance[4]}, '
                    f'Accuracy Tasks: {performance[5]:.3f}, Accuracy Subjects: {performance[6]:.3f}, '
                    f'Balanced Accuracy Tasks: {performance[7]:.3f}, Balanced Accuracy Subjects: {performance[8]:.3f}\n')
    avg_loss = np.mean([performance[0] for performance in fold_performance])
    avg_f1 = np.mean([performance[3] for performance in fold_performance])
    if folds > 1:
        accuracy = 0.0
        for performance in fold_performance:
            accuracy += performance[7]
        avg_accuracy = accuracy / folds
        print(f'Average Loss: {avg_loss:.3f}, Average F1: {avg_f1}')
        with open(f'{name}_validation_log.txt', 'a') as f:
            f.write(f'Average Loss: {avg_loss:.3f}, Average F1: {avg_f1}, Average Accuracy: {avg_accuracy:.3f}\n')

# ------------------------------------------ CREATE DATALOADER ------------------------------------------ #

def create_data_loader(train_tensor, val_tensor, batch_size, num_workers):
    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              prefetch_factor=3, persistent_workers=True)
    val_loader = DataLoader(val_tensor, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            prefetch_factor=3, persistent_workers=True)
    return train_loader, val_loader

# ------------------------------------------ FIND BEST FOLD ------------------------------------------ #

def find_minum_loss(filename):
    with open(filename, "r") as file:
        lines = file.readlines()
    
    min_loss = float('inf')
    best_fold = None

    for line in lines:
        if "Loss:" in line and "Average" not in line:
            parts = line.split()
            fold_num = int(parts[1])
            loss_value = float(parts[4].strip(','))

            if loss_value < min_loss:
                min_loss = loss_value
                best_fold = fold_num
    
    return best_fold

# ------------------------------------------ JointCrossEntropyLoss ------------------------------------------ #

class JointCrossEntropyLoss(nn.Module):
    def __init__(self, lamd : float = 0.6) -> None:
        super().__init__()
        self.lamd = lamd

    def forward(self, out, label):
        end_out = out[0]
        branch_out = out[1]
        end_loss = F.nll_loss(end_out, label)
        if isinstance(branch_out, list):
            branch_loss = [F.nll_loss(out, label).unsqueeze(0) for out in branch_out]
            branch_loss = torch.cat(branch_loss)
        else:
            branch_loss = F.nll_loss(branch_out, label)
        loss = self.lamd * end_loss + (1 - self.lamd) * torch.sum(branch_loss)
        return loss

# ------------------------------------------ VALIDATE ------------------------------------------ #

def validate(model, val_loader, criterion_tasks, device):
    model.eval()
    all_preds_tasks = []
    all_preds_subjects = []
    all_labels_tasks = []
    all_labels_subjects = []
    with torch.no_grad():
        for x_raw_batch, labels_batch, subjects_batch in val_loader:
            x_raw_batch, labels_batch, subjects_batch = x_raw_batch.to(device), labels_batch.to(device), subjects_batch.to(device)
            output_tasks, output_subjects = model(x_raw_batch)

            if isinstance(criterion_tasks, JointCrossEntropyLoss):
                output_tasks = output_tasks[0]
                output_subjects = output_subjects[0]
            
            _, preds_tasks = torch.max(output_tasks, 1)
            all_preds_tasks.extend(preds_tasks.cpu().numpy())
            all_labels_tasks.extend(labels_batch.cpu().numpy())

            _, preds_subjects = torch.max(output_subjects, 1)
            all_preds_subjects.extend(preds_subjects.cpu().numpy())
            all_labels_subjects.extend(subjects_batch.cpu().numpy())

        
        f1_tasks = f1_score(all_labels_tasks, all_preds_tasks, average=None)
        f1_subjects = f1_score(all_labels_subjects, all_preds_subjects, average=None)
        accuracy_tasks = accuracy_score(all_labels_tasks, all_preds_tasks)
        accuracy_subjects = accuracy_score(all_labels_subjects, all_preds_subjects)
        balanced_accuracy_tasks = balanced_accuracy_score(all_labels_tasks, all_preds_tasks)
        balanced_accuracy_subjects = balanced_accuracy_score(all_labels_subjects, all_preds_subjects)
        kappa_tasks = cohen_kappa_score(all_labels_tasks, all_preds_tasks)
        
    return f1_tasks.tolist(), f1_subjects.tolist(), accuracy_tasks, accuracy_subjects, \
        balanced_accuracy_tasks, balanced_accuracy_subjects, kappa_tasks
