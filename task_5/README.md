# Data Science HW5

## Create the Environment
```
Numpy == 1.21.5
Scipy == 1.7.3
Pytorch == 1.12.1
```

## Run

```
python3 main.py --input_file target_nodes_list.txt --data_path ./data/data.pkl --model_path saved-models/gcn.pt
```

## TODO
* attacker.py
  
  * def find_adj_node: adj list 轉成 相鄰 node index 的 list
  * def compare_two_list: 比較 target 跟不同 label node 的相鄰 node index 的 list
  * def get_score_and_chaneged_node: 用上一個 function return 的 list ，算一個分數和找出要改動的 node

