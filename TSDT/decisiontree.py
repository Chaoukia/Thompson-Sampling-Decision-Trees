import numpy as np
from scipy.special import xlogy
from copy import copy, deepcopy
from nltk import Tree


class DecisionTreeNode:
    """
    Description
    -----------------------
    Class describing nodes of Decision Trees.
    
    Class Attributes:
    -----------------------
    count   : Int, counter of the number of already created nodes. Serves as an id for each node.
    id_node : Dict: - keys   : Int, id of the node.
                    - values : DecisionTreeNode, the corresponding node.
    
    """
    
    count=0
    id_node = {}
    
    def __init__(self, attributes_categories, K, parent=None, attribute_value=None, n_samples=0, alpha=1, beta=1, min_samples=2):
        """
        Description
        -----------------------
        Constructor of class DecisionTreeNode.

        Attributes & Parameters
        -----------------------
        attributes_categories   : Dict: - keys   : Indices of the attributes to be considered for potential splits within the node.
                                        - values : Number of categories of each attribute.
        K                       : Int, the number of classes.
        parent                  : DecisionTreeNode object or None, the node's parent.
                                  None if the node is the tree's root.
        attribute_value         : tuple or None, the attribute and value that the node represents. None if the node is the root.
        n_samples               : Int, the number of samples seen in the node.
        alpha                   : Float > 0, alpha parameter of the Beta prior.
        beta                    : Float > 0, beta parameter of the Beta prior.
        min_samples             : Int, the minimum number of samples for the node (leaf) to be considered active.
        ALPHA                   : Float > 0, transformation of alpha used within the prior.
        BETA                    :Float > 0, transformation of beta used within the prior.
        id_node                 : Int, id of the created node.
        active                  : Boolean, whether the node is active or not.
        children                : Dict: - keys   : Attribute indices describing the branches below the node.
                                        - values : List of the childen nodes.
        statistics              : np.array of shape (n_attributes, attribute_category_max, K) where n_attributes is the number of remaining attributes for potential splits.
                                  statistics[i, j, k] stores the statistic nijk.
        statistics_alphas_betas : np.array of shape (n_attributes, attribute_category_max) where n_attributes is the number of remaining attributes for potential splits.
                                  statistics_alphas_betas[i, j] stores the ñij statistics.
        maps                    : Dict: - keys   : Indices of the remaining attributes for potential splits.
                                        - values : Enumerating indices, useful to map attributes to their indices in statistics and statistics_alphas_betas.
        list_indices            : List of indices of the remaining attributes for potential splits.
        range_indices           : range enumerating the remaining attributes for potential splits.
        n_classes               : np.array of shape (K,), the number of seen samples for each class.
        pred                    : Int in {1, ..., K}, the current estimated prediction of the node.
        value                   : Float, value of the node (entropy, gini or majority).
        expanded                : Boolean, whether node has been used in an expansion step.
        """
        
        self.attributes_categories = attributes_categories
        if self.attributes_categories:
            self.attribute_category_max = max(attributes_categories.values())
            
        else:
            self.attribute_category_max=1
            
        self.K = K
        self.id = DecisionTreeNode.count
        DecisionTreeNode.count += 1
        self.parent = parent
        self.attribute_value = attribute_value
        self.n_samples = n_samples
        self.alpha = alpha
        self.ALPHA = alpha**1
        self.beta = beta
        self.BETA = beta**1
        self.min_samples = min_samples
        self.value = -1/2
        self.active = (self.n_samples >= self.min_samples)
        self.children = {}
        self.statistics, self.statistics_alphas_betas = np.zeros((len(self.attributes_categories), self.attribute_category_max, self.K)), np.zeros((len(self.attributes_categories), self.attribute_category_max))
        self.maps = dict(zip(self.attributes_categories.keys(), range(len(self.attributes_categories))))
        self.list_indices, self.range_indices = list(self.attributes_categories.keys()), list(self.maps.values())
        self.n_classes = np.zeros(self.K)
        self.pred = 0
        
        self.id_node[self.id] = self
        self.expanded = False
        
    def update_statistics(self, X, y, update_alpha_beta=True):
        """
        Description
        -----------------------
        Update the node's nijk, ñij, alpha and beta statistics.
        
        Parameters
        -----------------------
        X                 : np.array, the current data point that is sorted into the leaf.
        y                 : Int, X's class.
        update_alpha_beta : Boolean, whether to update the statistics related to the potential children or not.
        
        Returns
        -----------------------
        """
        
        X_ = np.array(X)[self.list_indices]
            
        if update_alpha_beta:
            # Update the statistics related to the potential children nodes.
            self.statistics_alphas_betas[self.range_indices, X_] += (np.argmax(self.statistics[self.range_indices, X_, :], 1) == y)
            boolean = (self.pred == y)
            self.alpha += boolean
            self.ALPHA = self.alpha**1
            self.beta += 1 - boolean
            self.BETA = self.beta**1
            self.statistics[self.range_indices, X_, y] += 1
        
        # Update the statistics related to the current node.
        self.n_samples += 1
        self.n_classes[y] += 1
        self.pred = np.argmax(self.n_classes)
    
    def update_status(self):
        """
        Description
        -----------------------
        Update the status of the leaf, whether it is active or not.
        
        Parameters
        -----------------------
        
        Returns
        -----------------------
        """
        
        self.active = (self.n_samples >= self.min_samples)
            
    def set_parent(self, parent):
        """
        Description
        -----------------------
        Set the node's parent. 
        
        Parameters
        -----------------------
        parent : DecisionTreeNode, the node's parent.
        
        Returns
        -----------------------
        """
        
        self.parent = parent
        
    def add_child(self, child, index):
        """
        Description
        -----------------------
        Add child to the list of the node's children at the corresponding branch index.
        
        Parameters
        -----------------------
        child : DecisionTreeNode, the child node to be added.
        index : Int, the branch index where the child should be added.
        
        Returns
        -----------------------
        """
        
        try:
            self.children[index].append(child)
            
        except KeyError:
            self.children[index] = []
            self.children[index].append(child)
            
    def replace_child(self, index, i, child):
        """
        Description
        -----------------------
        Replace a child by another
        
        Parameters
        -----------------------
        index : Int, branch index.
        i     : Int, index of the child to replace.
        child : DecisionTreeNode, the child node to be added.
        
        Returns
        -----------------------
        """
        
        self.children[index][i] = child
        
    def set_attribute_value(self, attribute_value):
        """
        Description
        -----------------------
        Set the attribute and value that the node represents.
        
        Parameters
        -----------------------
        attribute_value : tuple or None, the attribute and value that the node represents. None if the node is the root.
        
        Returns
        -----------------------
        """
        
        self.attribute_value = attribute_value
        
    def update_value(self, metric='entropy'):
        """
        Description
        -----------------------
        Calculate the Entropy or Gini impurity of the node.
        
        Parameters
        -----------------------
        metric  : String in ['entropy', 'gini', 'majority'].
        
        Returns
        -----------------------
        """
        
        if self.n_samples == 0:
            return -10
        
        ratios = self.n_classes/self.n_samples
        if metric == 'entropy':
            self.value = xlogy(ratios, ratios).sum()
            
        elif metric == 'gini':
            self.value = (ratios**2).sum() - 1
            
        elif metric == 'majority':
            self.value = ratios.max() - 1
    
    def __copy__(self):
        """
        Description
        -----------------------
        Create a copy of the node.
        
        Parameters
        -----------------------
        
        Returns
        -----------------------
        New copy.
        """
        
        cls = self.__class__
        node = cls.__new__(cls)
        node.__dict__.update(self.__dict__)
        return node
    
    def __eq__(self, node_other):
        """
        Description
        -----------------------
        Make the comparison == between two nodes based on their value.
        
        Parameters
        -----------------------
        node_other : DecisionTreeNode, the node to compare with.
        
        Returns
        -----------------------
        True if value(node_current) == value(node_other)
        """
        
        return self.attribute_value == node_other.attribute_value
    
    def __ne__(self, node_other):
        """
        Description
        -----------------------
        Make the comparison != between two nodes based on their value.
        
        Parameters
        -----------------------
        node_other : DecisionTreeNode, the node to compare with.
        
        Returns
        -----------------------
        True if value(node_current) != value(node_other)
        """
        
        return self.attribute_value != node_other.attribute_value
    
    def __lt__(self, node_other):
        """
        Description
        -----------------------
        Make the comparison < between two nodes based on their value.
        
        Parameters
        -----------------------
        node_other : DecisionTreeNode, the node to compare with.
        
        Returns
        -----------------------
        True if value(node_current) < value(node_other)
        """
        
        return self.value < node_other.value
    
    def __le__(self, node_other):
        """
        Description
        -----------------------
        Make the comparison <= between two nodes based on their value.
        
        Parameters
        -----------------------
        node_other : DecisionTreeNode, the node to compare with.
        
        Returns
        -----------------------
        True if value(node_current) <= value(node_other)
        """
        
        return self.value <= node_other.value
    
    def __gt__(self, node_other):
        """
        Description
        -----------------------
        Make the comparison > between two nodes based on their value.
        
        Parameters
        -----------------------
        node_other : DecisionTreeNode, the node to compare with.
        
        Returns
        -----------------------
        True if value(node_current) > value(node_other)
        """
        
        return self.value > node_other.value
    
    def __ge__(self, node_other):
        """
        Description
        -----------------------
        Make the comparison >= between two nodes based on their value.
        
        Parameters
        -----------------------
        node_other : DecisionTreeNode, the node to compare with.
        
        Returns
        -----------------------
        True if value(node_current) >= value(node_other)
        """
        
        return self.value >= node_other.value
    
class DecisionTree:
    """
    Description
    -----------------------
    Class describing a decision tree.
    """
    
    def __init__(self, categories, K, nodes_dict=None, leaves=None, min_samples=2, metric='entropy'):
        """
        Description
        -----------------------
        Constructor of class DecisionTree.

        Attributes & Parameters
        -----------------------
        categories            : np.array of shape (n_attributes,), the number of categories of each attribute.
        K                     : Int, the number of classes.
        nodes_dict            : Dict: - keys   : id numbers of the nodes in the tree.
                                      - values : Int, index of the branch where to look for children (in the node's children dictionary), None if the node is a leaf.
                                None if leaves is None.
        leaves                : List or None, the list of the tree leaves.
                                If None, create a root that is the only tree leaf and populate the list of leaves with it.
        min_samples           : Int, the minimum number of samples for the node (leaf) to be considered active.
        metric                : String in ['entropy', 'gini'].
        attributes_categories : Dict: - keys   : Indices of the attributes.
                                      - values : Number of categories of each attribute.
        root                  : DecisionTreeNode, the tree's root.
        n_samples             : Int, the number of samples seen by the decision tree.
        value                 : Float, the information entropy of the tree, we initialize it to None.
        """
        
        self.categories = categories
        self.K = K
        self.attributes_categories = dict(zip(np.arange(len(categories)), categories))
        if leaves is None:
            root = DecisionTreeNode(self.attributes_categories, self.K, min_samples=min_samples)
            self.leaves = [root]
            self.nodes_dict = {root.id : None}
            
        else:
            self.leaves = leaves
            self.nodes_dict = nodes_dict
            
        self.min_samples = min_samples
        self.metric = metric
        self.root = self.retrieve_root()
        self.n_samples = 0
        self.value = None
        
    def retrieve_root(self):
        """
        Description
        -----------------------
        Retrieve the tree's root.

        Parameters
        -----------------------
        
        Returns
        -----------------------
        DecisionTreeNode, the root.
        """
        
        leaf = self.leaves[0]
        parent = leaf.parent
        while parent is not None:
            leaf = parent
            parent = leaf.parent
            
        return leaf
        
    def sort_data(self, X, y=None):
        """
        Description
        -----------------------
        Sort data point X into the corresponding leaf.

        Parameters
        -----------------------
        X : np.array, the data point to be sorted into a leaf.
        y : Int or None, the class of X. If None, then we do not update any statistics when sorting X into its corresponding leaf.
        
        Returns
        -----------------------
        node : DecisionTreeNode, the leaf where X is sorted.
        """
        
        node = self.root
        
        if y is not None:
            node.update_statistics(X, y)
            if self.nodes_dict[node.id] is None:      # If the node is a leaf, update its status and statistics.
                node.update_status()      # Update the status of the leaf (whether it is active or not).
        
        index = self.nodes_dict[node.id]
        while index is not None:
            value = X[index]
            node = node.children[index][value]
            if y is not None:
                node.update_statistics(X, y)
                if self.nodes_dict[node.id] is None:      # If the node is a leaf, update its status and statistics.
                    node.update_status()     # Update the status of the leaf (whether it is active or not).
                    
            index = self.nodes_dict[node.id]
            
        return node
    
    def get_leaves(self):
        """
        Description
        -----------------------
        Get the leaves.

        Parameters
        -----------------------
        
        Returns
        -----------------------
        leaves : List, the tree leaves.
        """
        
        return self.leaves
            
    def compute_weight(self, node, memo, fill=True):
        """
        Description
        -----------------------
        Compute the weight of a node, the estimator of P(X in node).

        Parameters
        -----------------------
        node : DecisionTreeNode, the considered node.
        memo : List, sums of the samples in siblings starting from node and going up to the root.
               If memo is empty, calculate these sums.
        fill : Boolean, whether to calculate the sums of samples and fill in the memo list or not.
        
        Returns
        -----------------------
        value : Float, the estimated weight.
        """
        
        if node.n_samples == 0:
            return 0
        
        if node.parent is None:
            return 1
        
        if not fill:
            return (node.n_samples/memo[0])*self.compute_weight(node.parent, memo[1:], False)
            
        else:
            n_samples_sum = 0
            for sibling in node.parent.children[node.attribute_value[0]]:
                n_samples_sum += sibling.n_samples
                
            memo.append(n_samples_sum)
            return (node.n_samples/n_samples_sum)*self.compute_weight(node.parent, memo, True)
        
    def update_value(self):
        """
        Description
        -----------------------
        Calculate the value of the tree.

        Parameters
        -----------------------
        
        Returns
        -----------------------
        value : Float, the tree value in UCT.
        """
        
        self.value = 0
        self.n_samples = 0
        # If the DT consists only of the root.
        if self.leaves[0].parent is None:
            self.leaves[0].update_value(self.metric)
            self.value += self.leaves[0].value
            self.n_samples += self.leaves[0].n_samples
            return
        
        for leaf in self.leaves:
            self.n_samples += leaf.n_samples
            
        memo_dict = {}        
        for leaf in self.leaves:
            leaf.update_value(self.metric)
            try:
                memo_dict[leaf.parent.id]
                weight = self.compute_weight(leaf, memo_dict[leaf.parent.id], fill=False)
                self.value += weight*leaf.value

            except KeyError:
                memo = []
                weight = self.compute_weight(leaf, memo, fill=True)
                self.value += weight*leaf.value
                # If leaf.n_samples == 0 then we wouldn't have filled the memo.
                if leaf.n_samples != 0:
                    memo_dict[leaf.parent.id] = memo
        
    def split(self, leaf, attribute):
        """
        Description
        -----------------------
        Split a leaf in the Decision Tree with respect to an attribute.

        Parameters
        -----------------------
        leaf      : DecisionTreeNode object, the leaf to split.
        attribute : Int, index of the attribute to split on.
        
        Returns
        -----------------------
        tree: the new DecisionTree that is split.
        """
        
        cls = self.__class__
        tree = cls.__new__(cls)
        attributes_categories = deepcopy(leaf.attributes_categories)
        attributes_categories.pop(attribute)
        nodes_dict = copy(self.nodes_dict)
        nodes_dict[leaf.id] = attribute

        for category in range(self.attributes_categories[attribute]):
            n_samples = leaf.statistics[leaf.maps[attribute], category, :].sum()
            n_samples_alpha = leaf.statistics_alphas_betas[leaf.maps[attribute], category]
            if n_samples < n_samples_alpha:
                raise ValueError('n_samples should be greater than n_samples_alpha')
                
            child = DecisionTreeNode(attributes_categories, self.K, parent=leaf, attribute_value=(attribute, category), n_samples=n_samples, 
                                     alpha = 1 + n_samples_alpha, beta = 1 + n_samples - n_samples_alpha)
            child.n_classes = copy(leaf.statistics[leaf.maps[attribute], category, :])
            
            #### Recently added.
            child.pred = np.argmax(child.n_classes)
            ####
            
            leaf.add_child(child, attribute)
            nodes_dict[child.id] = None
            
        leaves = []
        for leaf_ in self.leaves:
            if leaf_.id != leaf.id:
                leaves.append(leaf_)

        leaves = leaves + leaf.children[attribute]
        tree.__dict__.update(self.__dict__)
        tree.leaves = leaves
        tree.nodes_dict = nodes_dict
        return tree
            
    def predict(self, X, method='mc'):
        """
        Description
        -----------------------
        Predict the class of data point X.

        Attributes
        -----------------------
        X      : np.array, the data point to be classified.
        method : String in ['mc', 'nb']. 
                  - 'mc' for majority class.
                  - 'nb' for naive bayes.
        
        Returns
        -----------------------
        y_pred : Int, the predicted class of data point X.
        """
        
        leaf = self.sort_data(X)
        if method=='mc':
            return np.argmax(leaf.n_classes)
        
        elif method=='nb':
            return np.argmax(leaf.n_classes*np.prod([leaf.statistics[attribute][X[attribute]] for attribute in leaf.statistics], axis=0))
    
    def build_string_node(self, node):
        """
        Description
        --------------
        Build string representation (with parentheses) of the subtree starting from node.
        
        Parameters
        --------------
        node : DecisionTreeNode.
        
        Returns
        --------------
        string : String representation with parentheses of the policy as a tree with state its root.
        """
        
        string = ''
        if self.nodes_dict[node.id] is None:
            return string
        
        for child in node.children[self.nodes_dict[node.id]]:
            attribute, value = child.attribute_value
            string += '(X_' + str(attribute) + '=' + str(value) + ' ' + self.build_string_node(child) + ') '

        return string
    
    def build_string(self):
        """
        Description
        --------------
        Build string representation of the decision tree.
        
        Parameters
        --------------
        
        Returns
        --------------
        string : String representation with parentheses of the policy as a tree.
        """
        
        return '( ' + self.build_string_node(self.root) + ')'
    
    def plot_tree(self):
        """
        Description
        --------------
        Plot the decision tree.
        
        Parameters
        --------------
        
        Returns
        --------------
        nltk tree object, visualize the decision tree.
        """

        return Tree.fromstring(self.build_string())