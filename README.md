# Implementation of Breaking the Label-Structure Co-Corruption in Graph Partial Label Learning
This repository contains the reference code for the paper Breaking the Label-Structure Co-Corruption in Graph Partial Label Learning

## Demo for Tunning

```shell
python train.py --dataset_name Cora --partial-rate 0.8 --tunning
```

## Demo for Training

#### Candidate Labels with Uniform Noise

```shell
python train.py --dataset_name Cora --cls-lr 1 --cls-wd 5e-4 --alpha 0.6 --xi 0.1 --partial-rate 0.8
```

##### Candidate Labels with Neighborhood Noise

```shell
python train.py --dataset_name Cora --cls-lr 5e-1 --cls-wd 1e-4 --alpha 0.6 --xi 0.0 --k 4
```

##### PLCIFAR10G with Real-world Candidate Labels

```shell
python train.py --dataset_name plcifar10 --cls-lr 0.05 --cls-wd 1e-6 --alpha 0.3 --xi 0.5 --epoch 500 
```

## Requirements

* python 3.9.18

* pytorch 1.12.1

* torchvision 0.13.1

* pyg 2.3.0

* scikit-learn 1.3.0

* termcolor 2.1.0

* tqdm 4.65.0

* ruamel.yaml 0.18.6

* ray-tune 2.43.0

## Reference

* https://github.com/wwangwitsel/PLCIFAR10/tree/main
* https://github.com/YujieMo/SUGRL
