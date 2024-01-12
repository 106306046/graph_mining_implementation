import numpy as np
import scipy.sparse as sp
from torch.nn.modules.module import Module


class BaseAttack(Module):
    """Abstract base class for target attack classes.
    Parameters
    ----------
    model :
        model to attack
    nnodes : int
        number of nodes in the input graph
    device: str
        'cpu' or 'cuda'
    """

    def __init__(self, model, nnodes, device="cpu"):
        super(BaseAttack, self).__init__()
        self.surrogate = model
        self.nnodes = nnodes
        self.device = device

        self.modified_adj = None

    def attack(self, ori_adj, n_perturbations, **kwargs):
        """Generate perturbations on the input graph.
        Parameters
        ----------
        ori_adj : scipy.sparse.csr_matrix
            Original (unperturbed) adjacency matrix.
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        Returns
        -------
        None.
        """
        raise NotImplementedError()


class RND(BaseAttack):
    def __init__(self, model=None, nnodes=None, device="cpu"):
        super(RND, self).__init__(model, nnodes, device=device)

    def attack(
        self,
        ori_features: sp.csr_matrix,
        ori_adj: sp.csr_matrix,
        labels: np.ndarray,
        idx_train: np.ndarray,
        target_node: int,
        n_perturbations: int,
        **kwargs,
    ):
        """
        Randomly sample nodes u whose label is different from v and
        add the edge u,v to the graph structure. This baseline only
        has access to true class labels in training set
        Parameters
        ----------
        ori_features : scipy.sparse.csr_matrix
            Original (unperturbed) node feature matrix
        ori_adj : scipy.sparse.csr_matrix
            Original (unperturbed) adjacency matrix
        labels :
            node labels
        idx_train :
            node training indices
        target_node : int
            target node index to be attacked
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could be edge removals/additions.
        """

        print(f"number of pertubations: {n_perturbations}")
        modified_adj = ori_adj.tolil()
        row = ori_adj[target_node].todense().A1
        diff_label_nodes = [
            x for x in idx_train if labels[x] != labels[target_node] and row[x] == 0
        ]  # 取不同 label 且目前沒有 edge 的 node

        diff_label_nodes = np.random.permutation(
            diff_label_nodes
        )  # 把 diff_label_nodes 順序打亂

        if len(diff_label_nodes) >= n_perturbations:
            changed_nodes = diff_label_nodes[:n_perturbations]  # 取前 n_perturbations 個
            print("target:", target_node)
            print("changed_nodes", changed_nodes)
            modified_adj[target_node, changed_nodes] = 1
            modified_adj[changed_nodes, target_node] = 1
        else:
            changed_nodes = diff_label_nodes
            unlabeled_nodes = [
                x for x in range(ori_adj.shape[0]) if x not in idx_train and row[x] == 0
            ]
            unlabeled_nodes = np.random.permutation(unlabeled_nodes)
            changed_nodes = np.concatenate(
                [
                    changed_nodes,
                    unlabeled_nodes[: n_perturbations - len(diff_label_nodes)],
                ]
            )
            print("target:", target_node)
            print("changed_nodes", changed_nodes)
            modified_adj[target_node, changed_nodes] = 1
            modified_adj[changed_nodes, target_node] = 1
            pass

        self.modified_adj = modified_adj


class MyAttacker(BaseAttack):
    def __init__(self, model=None, nnodes=None, device="cpu"):
        super(MyAttacker, self).__init__(model, nnodes, device=device)

    def attack(
        self,
        ori_features: sp.csr_matrix,
        ori_adj: sp.csr_matrix,
        labels: np.ndarray,
        idx_train: np.ndarray,
        target_node: int,
        n_perturbations: int,
        **kwargs,
    ):
        print(f"number of pertubations: {n_perturbations}")
        modified_adj = ori_adj.tolil()
        row = ori_adj[target_node].todense().A1
        diff_label_nodes = [
            x for x in idx_train if labels[x] != labels[target_node]
        ]  # 取不同 label 且的 node

        # 取相鄰 node 中跟 target 最像的 node
        best_score = float("-inf")
        best_changed_0_node_list = []
        best_changed_1_node_list = []

        for node in diff_label_nodes:
            diff_node_adj = ori_adj[node].todense().A1
            (
                list_diff_in_target,
                list_same,
                list_diff_in_diff_node,
            ) = self.compare_two_list(
                self.find_adj_node(row), self.find_adj_node(diff_node_adj)
            )
            (
                score,
                changed_0_node_list,
                changed_1_node_list,
            ) = self.get_score_and_chaneged_node(
                list_diff_in_target, list_same, list_diff_in_diff_node, n_perturbations
            )
            if score > best_score:
                best_changed_0_node_list = changed_0_node_list
                best_changed_1_node_list = changed_1_node_list

        print("target:", target_node)
        print("changed_to_0_nodes", best_changed_0_node_list)
        print("changed_to_1_nodes", best_changed_1_node_list)

        if len(best_changed_0_node_list) != 0:
            modified_adj[target_node, best_changed_0_node_list] = 0
            modified_adj[best_changed_0_node_list, target_node] = 0
        if len(best_changed_1_node_list) != 0:
            modified_adj[target_node, best_changed_1_node_list] = 1
            modified_adj[best_changed_1_node_list, target_node] = 1

        self.modified_adj = modified_adj

    def find_adj_node(self, adj_list):
        adj_node = []
        for idx in range(0, len(adj_list)):
            if adj_list[idx] == 1:
                adj_node.append(idx)

        return adj_node

    def compare_two_list(self, list_target, list_diff_node):
        list_same = []
        list_diff_in_target = []
        list_diff_in_diff_node = []

        for item in list_diff_node:
            if item in list_target:
                list_same.append(item)
            else:
                list_diff_in_diff_node.append(item)

        list_diff_in_target = [item for item in list_target if item not in list_same]
        return list_diff_in_target, list_same, list_diff_in_diff_node

    def get_score_and_chaneged_node(
        self, list_diff_in_target, list_same, list_diff_in_diff_node, budget
    ):
        score = 0
        penalty = -1000
        changed_0_node_list = []
        changed_1_node_list = []
        same_score = len(list_same)
        diff_score = budget - len(list_diff_in_target) - len(list_diff_in_diff_node)

        if (
            diff_score < 0
        ):  # 可以改的 > budget -> 讓 score 很低 ＆ return 符合 budget 的 changed_node_list
            score = same_score - abs(diff_score)
            changed_1_node_list = list_diff_in_diff_node[:budget]
            if len(changed_1_node_list) < budget:
                changed_0_node_list = list_diff_in_target[
                    : budget - len(changed_1_node_list)
                ]
        else:  # 可以改的 <= budget
            score = same_score - abs(diff_score)
            changed_0_node_list = list_diff_in_target
            changed_1_node_list = list_diff_in_diff_node

        return score, changed_0_node_list, changed_1_node_list
