import sys
import pandas as pd

min_support = sys.argv[1]
input_fileName = sys.argv[2]
output_fileName = sys.argv[3]

print('min support: ', min_support)
print('input_fileName: ', input_fileName)
print('output_fileName: ', output_fileName)

# load data

df = pd.read_csv(input_fileName, sep=' ', header=0)
df.columns = ['tx']

min_support = min_support*len(df)

# node


class node:
    def __init__(self, item, frequent, parent, child=[]):
        self.item = item
        self.frequent = frequent
        self.parent = parent
        self.child = child

    def list_child_item(self):
        child_item_list = []

        for child in self.child:
            child_item_list.append(child.item)

        return child_item_list

    def search_child(self, item):
        for child in self.child:
            if item == child.item:
                return child
        return None

    def printTree(self, level=0):
        print('\t' * level, '[level: ', level, 'item: ',
              self.item, 'fre: ', self.frequent, ']')
        for child in self.child:
            child.printTree(level+1)

# construct 1-itemset


itemset_1 = pd.DataFrame(columns=['item', 'frequent', 'head_list'])

for index, row in df.iterrows():

    tx = row['tx'].split(",")

    for item in tx:

        if item in itemset_1['item'].values:
            itemset_1.loc[itemset_1['item'] == item, 'frequent'] += 1
        else:
            itemset_1.loc[len(itemset_1.index)] = [item, 1, []]

# sort by frequent

itemset_1 = itemset_1.sort_values('frequent', ascending=False)
itemset_1 = itemset_1.set_index('item')

# 去掉小於min support

for i in range(0, len(itemset_1)):

    if itemset_1.iloc[i].frequent < min_support:
        itemset_1 = itemset_1.iloc[0:i]
        break
# 依據 itemset_1 排序 tx

sorted_item = itemset_1.index

for index, row in df.iterrows():

    tx = row['tx'].split(",")

    sorted_tx = []

    for item in sorted_item:

        if item in tx:
            sorted_tx.append(item)

    row['tx'] = sorted_tx

# construct tree

root = node('root', None, None, [])

for tx in df['tx']:

    search_node = root

    for item in tx:
        # 在目前search_node的child node有此item就+1
        # 沒有就新增 childe的node

        child_node = search_node.search_child(item)

        if (child_node != None):
            child_node.frequent += 1
        else:
            # new node
            child_node = node(item, 1, search_node, [])
            # connect to header list
            itemset_1.loc[item].head_list.append(child_node)
            # connect to tree
            search_node.child.append(child_node)

        search_node = child_node
print('Tree:', root.printTree())

# Construct Mining FP Tree
pattern = []

for mining in itemset_1.index:

    sub_root = node('root', 1, None, [])
    total_frequent = {}
    path_and_fre = []
    sub_header = []
    for mining_node in itemset_1.loc[mining].head_list:
        # construct a subtree which leaf is mining

        # get list from leaf to node

        path = []
        mining_node_frequent = mining_node.frequent
        path.append(mining_node.item)

        mining_node_parent = mining_node.parent

        while mining_node_parent.item != 'root':

            path.append(mining_node_parent.item)

            if mining_node_parent.item in total_frequent:
                total_frequent[mining_node_parent.item] += mining_node_frequent
            else:
                total_frequent[mining_node_parent.item] = 1

            mining_node_parent = mining_node_parent.parent

        path.reverse()
        path_and_fre.append([path, mining_node_frequent])

    discharge_node = [k for k, v in total_frequent.items() if v < min_support]

    for pair in path_and_fre:

        search_node = sub_root

        path = pair[0]
        fre = pair[1]

        for item in path:

            if item not in discharge_node:
                child_node = search_node.search_child(item)

                if (child_node != None):
                    child_node.frequent += mining_node_frequent
                else:
                    child_node = node(item, fre, search_node, [])
                    if item == mining:
                        sub_header.append(child_node)
                    search_node.child.append(child_node)

                search_node = child_node

    for header in sub_header:
        path = []

        search_node = header
        fre = header.frequent

        while search_node.item != 'root':
            path.append(search_node.item)
            search_node = search_node.parent

        path.sort()
        pattern.append([path, fre/len(df)])

    for pa in pattern:
        search_pattern = pattern.copy()
        search_pattern.remove(pa)
        for pa_2 in search_pattern:
            if set(pa[0]).issubset(pa_2[0]):
                pa[1] += pa_2[1]

# output

with open(output_fileName, 'w') as f:
    for pa in pattern:
        f.write(pa[0], ':', pa[1], '\n')
