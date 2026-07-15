import sys
import argparse
from utils import create_tensors_subjects, find_minum_loss, validate, \
    load_normalizations, available_paradigm, available_network, \
        network_factory_methods, JointCrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader
import json
import numpy as np
import torch
import torch.nn as nn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_set', type=str, default='Test_Sets/test_2b.npz')
    parser.add_argument("--name_model", type=str, choices=available_network, default='MIRACLE')
    parser.add_argument('--saved_path', type=str, default='Weights_Models/Results_2b')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=157)
    parser.add_argument('--paradigm', type=str, choices=available_paradigm, default='Cross')
    parser.add_argument('--num_workers', type=int, default=5)
    args = parser.parse_args()
    
    data_test_tensors, labels_test_tensors, subjects_test_tensors = create_tensors_subjects(args.test_set)
    
    final_results = []
    f1_list_tasks, accuracy_list_tasks, balanced_accuracy_list_tasks = [], [], []
    f1_list_subjects, accuracy_list_subjects, balanced_accuracy_list_subjects = [], [], []
    kappa_tasks_list = []
    
    ###### DA RIVEDERE #######
    num_subjects=len(data_test_tensors)
    
    for subject in range(num_subjects):
        data, labels, subjects = data_test_tensors[subject], labels_test_tensors[subject], subjects_test_tensors[subject]
        
        mean, std = load_normalizations(f'{args.saved_path}/{args.name_model}')
        data = (data - mean)/std
        
        dataset = TensorDataset(data, labels, subjects)
        test_loader = DataLoader(dataset, batch_size=256, num_workers=args.num_workers)

        best_fold = find_minum_loss(f'{args.saved_path}/{args.name_model}_seed{args.seed}_validation_log.txt')
        
        model = (
            network_factory_methods[args.name_model](model_name_prefix=f'{args.saved_path}/{args.name_model}_seed{args.seed}',
                num_classes=len(np.unique(labels)), subjects=num_subjects, samples=data.shape[3], channels=data.shape[2])
        )
        model.to(args.device)
        model.load_state_dict(torch.load(f'{args.saved_path}/{args.name_model}_seed{args.seed}_best_model_fold{best_fold}.pth', map_location=args.device))

        if (args.name_model == "MSVTNet" or args.name_model == "MIRACLE"):
            criterion_tasks = JointCrossEntropyLoss()
        else:
            criterion_tasks = nn.CrossEntropyLoss()

        (val_f1_tasks, 
        val_f1_subjects, val_accuracy_tasks, 
        val_accuracy_subjects, val_balanced_accuracy_tasks, 
        val_balanced_accuracy_subjects, val_kappa_tasks) = validate(model, test_loader, criterion_tasks, device=args.device)

        f1_list_tasks.append(val_f1_tasks), accuracy_list_tasks.append(val_accuracy_tasks), balanced_accuracy_list_tasks.append(val_balanced_accuracy_tasks)
        f1_list_subjects.append(val_f1_subjects), accuracy_list_subjects.append(val_accuracy_subjects), balanced_accuracy_list_subjects.append(val_balanced_accuracy_subjects)
        kappa_tasks_list.append(val_kappa_tasks)
        
        final_results.append({'Subject': subject+1, 
                            'F1 Score Tasks': val_f1_tasks, 'Accuracy Tasks': val_accuracy_tasks, 'Balanced Accuracy Tasks': val_balanced_accuracy_tasks,
                            'Accuracy Subjects': val_accuracy_subjects, 'Balanced Accuracy Subjects': val_balanced_accuracy_subjects, 
                            'Kappa Tasks': val_kappa_tasks})
    
    final_results.append(
        {f"Average": {"F1 Score Tasks": np.mean(f1_list_tasks, axis=0).tolist(), "Accuracy Tasks": np.mean(accuracy_list_tasks), "Balanced Accuracy Tasks": np.mean(balanced_accuracy_list_tasks),
                    "Accuracy Subjects": np.mean(accuracy_list_subjects), "Balanced Accuracy Subjects": np.mean(balanced_accuracy_list_subjects), 
                    'Kappa Tasks': np.mean(kappa_tasks_list)}}
    )

    with open(f'{args.saved_path}/Final_results_{args.name_model}_seed{args.seed}.json', 'w') as f:
        json.dump(final_results, f, indent=4)

    print('All subject test results have been saved.')
    
    sys.exit()
