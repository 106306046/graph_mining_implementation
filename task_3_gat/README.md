# Data Science HW3

## TODO
* model.py
  * implement GAT from pytorch
* train.py
  * find the best hidden size in [16, 32, 64, 128, 256, 512, 1024] and number of heads in [2, 3, 4, 5]

## Run code

```python

python3 train.py --es_iters 30 --epochs 300 --use_gpu

```

## Dataset
* Unknown graph data
  * Label Split:
    * Train: 60, Valid: 500, Test: 1000
* File name description
```
  dataset
  │   ├── private_features.pkl # node feature
  │   ├── private_graph.pkl # graph edges
  │   ├── private_num_classes.pkl # number of classes
  │   ├── private_test_labels.pkl # X
  │   ├── private_test_mask.pkl # nodes indices of testing set
  │   ├── private_train_labels.pkl # nodes labels of training set
  │   ├── private_train_mask.pkl # nodes indices of training set
  │   ├── private_val_labels.pkl # nodes labels of validation set
  │   └── private_val_mask.pkl # nodes indices of validation set
```