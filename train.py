from data_unit.utils import mkdir_if_missing
from dataset.data_utils import generate_candidate_labels, get_dataset, init_seed, train_val_test_split
import time
import torch
import numpy as np
import os
import argparse
from termcolor import cprint

from models.classifier import Classifier
import ray
from ray import tune
from functools import partial
from models.presentation import ModelN
from models.structure import ModelE

def main(args):
    # ===================================================#
    init_seed(0)
    running_device = "cpu" if args.gpu is None \
    else torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    # ===================================================#
    cprint("## Loading Dataset ##", "yellow")
    
    dataset_kwargs = {}
    data, adj_list, x_list, nb_list = get_dataset(args, './', running_device)
    
    label = nb_list[0]
    nb_feature = nb_list[1]
    nb_classes = nb_list[2]
    nb_nodes = nb_list[3]
    feature_X = x_list[0].to(running_device)
    
    A_I_normal = adj_list[0].to(running_device)
    A = adj_list[1]
    A_I = adj_list[2]
    plabels = generate_candidate_labels(label, partial_rate=args.partial_rate, 
                                        k=args.k, adj_matrix=A,
                                        dataset=args.dataset_name)
    if args.dataset_name == 'plcifar10':
        label = data.ground_truth
        train_indices, val_indices, test_indices = data.train_idx, data.val_idx, data.test_idx
        train_indices, val_indices, test_indices = train_indices.numpy(), val_indices.numpy(), test_indices.numpy()
    else:
        train_indices, val_indices, test_indices = train_val_test_split(len(plabels), 0.1, 0.1)
    print(f'Labels Mean: {plabels[train_indices].sum(1).mean():.2f}')
    
    if args.results_dir is not None:
        results_dir = f'{args.results_dir}/DC/{args.dataset_name}_{args.partial_type}{args.partial_rate}_{args.seed}'
        mkdir_if_missing(f'{results_dir}')
    else:
        results_dir = None
    
    init_seed(args.seed)
    start_time = time.time()
    exn_yaml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models/args.yaml")
    
    explainerN = ModelN(dataset_name=args.dataset_name, yaml_path=exn_yaml_path,
                        running_device=running_device, 
                        feature_X=feature_X, A_I_normal=A_I_normal,
                        nb_feature=nb_feature, nb_nodes=nb_nodes,
                        results_dir=results_dir, label_matrix=plabels, idx_train=train_indices)
    model, embs = explainerN.train()
    
    explainerE = ModelE(epochs=100, 
                           target_logits=feature_X, 
                           features=embs, 
                           label_matrix=plabels, idx_train=train_indices,
                           adj=A_I,
                           lr=0.001,
                           results_dir=results_dir,
                           plabels=plabels)
    _, edge_mask = explainerE.train(
        model=model,
        x=feature_X,
        A_I=A_I,
        en_args=explainerN.args,
    )

    classifier = Classifier(in_channels=feature_X.shape[1], hidden_channels=64, feature_dim=embs.shape[1], 
                            classes=nb_classes, 
                            features=feature_X, edge_index=A_I._indices(),
                            plabels=plabels, groundtruth=label, 
                            edge_weight=A_I._values(), explain_features=embs, explain_weight=edge_mask, 
                            bias=True, self_loop=True, relu=True, dropout=0.5, results_dir=results_dir)
    acc_test, acc_val = classifier.train(epochs=args.epoch,
                                         idx_train=train_indices,
                                         idx_val=val_indices,
                                         idx_test=test_indices,
                                         lr=args.cls_lr,
                                         wd=args.cls_wd,
                                         momentum=0.99,
                                         device=running_device,
                                         alpha=args.alpha,
                                         xi=args.xi)
    end_time = time.time()
    print('run_time:', end_time - start_time) 
    
    return acc_test.item(), acc_val.item()


def param_search(config, args):
    args.cls_lr = config["lr"] if "lr" in config else args.cls_lr
    args.cls_wd = config["wd"] if "wd" in config else args.cls_wd
    args.alpha = config["alpha"] if "alpha" in config else args.alpha
    args.xi = config["xi"] if "xi" in config else args.xi
    args.gpu = str(0)
    acc_val_list = []
    acc_test_list = []
    for seed in [0, 1, 2, 3, 4]:
        args.seed = seed
        print(f"------------------seed{args.seed}------------------")
        print(args)
        acc_test, acc_val = main(args=args)
        acc_val_list.append(acc_val)
        acc_test_list.append(acc_test)
    print("mean ± std: {:.2f}±{:.2f}".format(np.mean(acc_test_list) * 100, np.std(acc_test_list) * 100))
    return {"acc_val": np.mean(acc_val_list) * 100, 'acc_test': np.mean(acc_test_list) * 100, 'std_test': np.std(acc_test_list) * 100}
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='CiteSeer', 
                        choices=['Cora', 'CiteSeer', 'PubMed', 'Photo', 'Computers', 'ogbn-arxiv', 'plcifar10'], type=str)
    parser.add_argument('--partial-rate', default=None, type=float)
    parser.add_argument('--k', default=None, type=int)
    parser.add_argument('--results-dir', default=None, type=str, help='directory to save results when not None')
    
    parser.add_argument('--cls-lr', default=1e-2, type=float)
    parser.add_argument('--cls-wd', default=5e-4, type=float)
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--xi', default=1.0, type=float)
    parser.add_argument('--epoch', default=200, type=float)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default='7', type=str, help='which gpu(s) can be used for training')
    parser.add_argument("--tunning", action="store_true")
    args = parser.parse_args()
    
    if args.tunning:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
        ray.init(num_gpus=4)
        config = {
            "lr": tune.grid_search([1, 5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3]),
            "wd": tune.grid_search([5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6]),
            "alpha": tune.grid_search([0, 0.3, 0.6, 0.9]),
            "xi": tune.grid_search([0.0, 0.5, 1.0, 1.5, 2.0]),
        }
        analysis = tune.run(
            partial(param_search, args=args), 
            config=config,
            resources_per_trial={"gpu":1},
            )
        print("Best config:", analysis.get_best_config(metric="acc_val", mode="max"))
        best_trial = analysis.get_best_trial(metric="acc_val", mode="max")
        print("Final results: {:.2f}±{:.2f}".format(best_trial.last_result["acc_test"], best_trial.last_result["std_test"]))
    else:
        acc = []
        for seed in [0, 1, 2, 3, 4]:
            args.seed = seed
            print(f"------------------seed{args.seed}------------------")
            print(args)
            acc_test, _ = main(args=args)
            acc.append(acc_test)
        print("mean ± std: {:.2f}±{:.2f}".format(np.mean(acc) * 100, np.std(acc) * 100))
        