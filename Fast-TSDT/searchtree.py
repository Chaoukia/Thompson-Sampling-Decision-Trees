from decisiontree import *
from itertools import chain
import heapq
from scipy.stats import norm
from time import time


class SearchTreeNode:
    """
    Description
    -----------------------
    Class describing a node of the Search Tree.
    """
    
    def __init__(self, tree, depth, parent=None, min_samples=2, to_expand=True, lambd=0, mu=None, sigma=None, gamma=0.5):
        """
        Description
        -----------------------
        Constructor of class SearchTreeNode.

        Attributes & Parameters
        -----------------------
        tree                     : DecisionTree represented by the node in the Search Tree of MCTS.
        parent                   : SearchTreeNode, the parent node. None if the node represents a root tree.
        min_samples              : Int, the minimum number of samples required to split a leaf.
        to_expand                : Boolean, whether the current SearchTreeNode is to be expanded. Useful for prepruning.
        lambd                    : Float in ]0, 1[, the penalty term penalising complex trees.
        mu                       : Float, mean of the normal approximation of the prior.
        sigma                    : Float, std of the normal approximation of the prior.
        gamma                    : Float > 0 , the exponent used to scale the std of the posterior distribution.
        children                 : Dict : - keys   : Int, ids of the leaves that are split.
                                                     We will use 1/2 as a key to the child representing the same node (useful for prepruning).
                                          - values : List, list of children SearchTree nodes.
        n_visits                 : Int, number of visits of the current SearchTreeNode.
        fully_expanded           : Boolean, whether the SearchTree node is fully expanded or not.
        leaves_active            : List of DecisionTreeNode objects, the remaining leaves to split until the SearchTree node becomes fully expanded.
                                   More precisely, it is a priority queue where the first element is the leaf with minimum value, i.e with the highest entropy, and
                                   indeed we want to prioritise splitting leaves with higher entropy.
        activated                : Boolean, whether the leaves have been activated or not.
        """
        
        self.tree = tree
        self.depth = depth
        self.parent = parent
        self.min_samples = min_samples
        self.to_expand = to_expand
        self.lambd = lambd
        self.mu = mu
        self.sigma = sigma
        self.gamma = gamma
        self.children = {}
        self.n_visits = 0
        self.fully_expanded = False
        self.leaves_active = []
        self.activated = False
                
    def update_visits(self):
        """
        Description
        -----------------------
        Increment the number of visits of the current SearchTreeNode.

        Parameters
        -----------------------
        
        Returns
        -----------------------
        """
        
        self.n_visits += 1
                
    def activate(self, leaf):
        """
        Description
        -----------------------
        Put a leaf in the leaves_active list.

        Parameters
        -----------------------
        leaf : DecisionTreeNode, the leaf to activate. It needs to be a previously inactive leaf.
        
        Returns
        -----------------------
        """
        
        heapq.heappush(self.leaves_active, leaf)
        
    def activate_all(self):
        """
        Description
        -----------------------
        Activate all active leaves.

        Parameters
        -----------------------
        
        Returns
        -----------------------
        """
        
        for leaf in self.tree.leaves:
            if leaf.active:
                self.activate(leaf)
                
        self.activated = True
        
    def update_statistics(self, X, y):
        """
        Description
        -----------------------
        Update the statistics of the decision tree represented by the SearchTree node.

        Parameters
        -----------------------
        X : np.array, a data point.
        y : Int, X's class.
        
        Returns
        -----------------------
        leaf    : DecisionTreeNode, the leaf where X is sorted.
        """
        
        return self.tree.sort_data(X, y)
    
    def update_mu_sigma(self):
        """
        Description
        -----------------------
        Update the mean and variance of the normal approximation of the Beta prior. Used when the Search Node has no children.

        Parameters
        -----------------------
        
        Returns
        -----------------------
        """
        
        # If the DT consists only of the root.
        if self.tree.leaves[0].parent is None:
            self.mu, self.sigma = approximate_beta(self.tree.leaves[0].ALPHA, self.tree.leaves[0].BETA, self.gamma)
            return
        
        self.mu, self.sigma = 0, 0
        memo_dict = {}
        for leaf in self.tree.leaves:
            mu, sigma = approximate_beta(leaf.ALPHA, leaf.BETA, self.gamma)
            try:
                weight = self.tree.compute_weight(leaf, memo_dict[leaf.parent.id], fill=False)
                self.mu += weight*mu
                self.sigma += (weight*sigma)**2

            except KeyError:
                memo = []
                weight = self.tree.compute_weight(leaf, memo, fill=True)
                self.mu += weight*mu
                self.sigma += (weight*sigma)**2
                # If leaf.n_samples == 0 then we wouldn't have filled the memo.
                if leaf.n_samples != 0:
                    memo_dict[leaf.parent.id] = memo
                    
        self.mu -= self.lambd*self.depth
        self.sigma = np.sqrt(self.sigma)**self.gamma
        
    def replace_mu_sigma(self, mu, sigma):
        """
        Description
        -----------------------
        Update the mean and variance of the normal approximation of the prior (max distribution over the children's priors).

        Parameters
        -----------------------
        
        Returns
        -----------------------
        """
        
        self.mu, self.sigma = mu, sigma
        
    def expand(self, leaf, thresh, thresh_tree):
#    def expand(self, leaf, thresh, thresh_mu, thresh_sigma):
        """
        Description
        -----------------------
        Expand the SearchTree node by splitting the tree at the given leaf with respect to all available attributes.

        Parameters
        -----------------------
        leaf   : DecisionTreeNode, The chosen leaf to split.
        thresh : Float, if leaf.value >= -thresh then leaf is considered pure.
        
        Returns
        -----------------------
        """
        
        if leaf.n_samples < self.min_samples:
            raise Exception("Not enough samples to split the leaf")
            
        if not self.to_expand:
            raise Exception("This node should not be expanded")
            
        if leaf.value >= -thresh:
            # Do not expand with respect to an approximately pure leaf, this is part of prepruning, it is supposed to speed up exploration.
            self.fully_expanded = True
            return None
        
        try:
            self.children[1/2][0].update_mu_sigma()
            mu_max, sigma_max = self.children[1/2][0].mu, self.children[1/2][0].sigma
            self.children[1/2][0].tree.update_value()
            if self.children[1/2][0].tree.value > -thresh_tree:
#            if (self.children[1/2][0].mu > thresh_mu) and (self.children[1/2][0].sigma < thresh_sigma):
                return self.children[1/2][0]
            
        except KeyError:
            self_copy = SearchTreeNode(self.tree, self.depth, parent=self, min_samples=self.min_samples, to_expand=False, lambd=self.lambd, gamma=self.gamma)
            self_copy.update_mu_sigma()
            mu_max, sigma_max = self_copy.mu, self_copy.sigma
            self_copy.tree.update_value()
            if self_copy.tree.value > -thresh_tree:
#            if (self_copy.mu > thresh_mu) and (self_copy.sigma < thresh_sigma):
                return self_copy
            
            self.children[1/2] = [self_copy]
            
        params = []    # params stores (mu, sigma) of all children for a future backpropagation step.
        self.children[leaf.id] = []    # List of children induced by splitting the leaf.
        
        # Update mu and sigma of the already existing children and add them to params for the backpropagation step.
        for child in chain.from_iterable(self.children.values()):
            child.update_mu_sigma()
            if child.mu > mu_max:
                mu_max, sigma_max = child.mu, child.sigma
                
            child.tree.update_value()
            if child.tree.value > -thresh_tree:
#            if (child.mu > thresh_mu) and (child.sigma < thresh_sigma):
                return child
            
            params.append((child.mu, child.sigma))
        
        if not leaf.expanded:
            for attribute in leaf.attributes_categories:
                tree = self.tree.split(leaf, attribute)
                to_expand = True
                if not tree.attributes_categories:
                    to_expand = False
                    
                searchtree_node = SearchTreeNode(tree, self.depth + 1, parent=self, min_samples=self.min_samples, to_expand=to_expand, lambd=self.lambd, gamma=self.gamma)
                searchtree_node.update_mu_sigma()
                if searchtree_node.mu > mu_max:
                    mu_max, sigma_max = searchtree_node.mu, searchtree_node.sigma
                    
                searchtree_node.tree.update_value()
                if searchtree_node.tree.value > -thresh_tree:
#                if (searchtree_node.mu > thresh_mu) and (searchtree_node.sigma < thresh_sigma):
                    return searchtree_node
                
                params.append((searchtree_node.mu, searchtree_node.sigma))
                self.children[leaf.id].append(searchtree_node)
                
            leaf.expanded = True
            
        else:
            for attribute in leaf.attributes_categories:
                tree = DecisionTree(self.tree.categories, self.tree.K, min_samples=self.tree.min_samples)
                tree.__dict__.update(self.tree.__dict__)
                leaves = []
                nodes_dict = copy(self.tree.nodes_dict)
                nodes_dict[leaf.id] = attribute
                for leaf_ in self.tree.leaves:
                    if leaf_.id != leaf.id:
                        leaves.append(leaf_)
                        
                for child in leaf.children[attribute]:
                    
                    ###### This block of code should also be added to UCDT, it explains a big loss of data that can occur!
                    category = child.attribute_value[1]
                    n_samples = leaf.statistics[leaf.maps[attribute], category, :].sum()
                    n_samples_alpha = leaf.statistics_alphas_betas[leaf.maps[attribute], category]
                    child.n_samples = n_samples
                    child.alpha = 1 + n_samples_alpha
                    child.beta = 1 + n_samples - n_samples_alpha
                    child.n_classes = copy(leaf.statistics[leaf.maps[attribute], category, :])
                    ######
                    
                    nodes_dict[child.id] = None

                leaves = leaves + leaf.children[attribute]
                tree.leaves = leaves
                tree.nodes_dict = nodes_dict
                to_expand = True
                if not tree.attributes_categories:
                    to_expand = False

                searchtree_node = SearchTreeNode(tree, self.depth + 1, parent=self, min_samples=self.min_samples, to_expand=to_expand, lambd=self.lambd, gamma=self.gamma)
                searchtree_node.update_mu_sigma()
                if searchtree_node.mu > mu_max:
                    mu_max, sigma_max = searchtree_node.mu, searchtree_node.sigma
                    
                searchtree_node.tree.update_value()
                if searchtree_node.tree.value > -thresh_tree:
#                if (searchtree_node.mu > thresh_mu) and (searchtree_node.sigma < thresh_sigma):
                    return searchtree_node
                
                params.append((searchtree_node.mu, searchtree_node.sigma))
                self.children[leaf.id].append(searchtree_node)
                            
        # Update mu and sigma for the normal approximation of the maximum distribution of the children's priors.
        self.replace_mu_sigma(mu_max, sigma_max)
        if not self.leaves_active:
            self.fully_expanded = True
            
    def sample(self):
        """
        Description
        -----------------------
        Sample from the normal approximation of the prior N(mu, sigma).
        
        Parameters
        -----------------------
        
        Returns
        -----------------------
        """
        
        return norm.rvs(loc=self.mu, scale=self.sigma)
        
    def __lt__(self, node_other):
        """
        Description
        -----------------------
        Make the comparison < between two SearchTree nodes based on their negative mu.
        This might seem confusing, but it is useful in order to use a heapq of SearchTreeNodes where
        the first element is the SearchTreeNode with maximum mu.
        
        Parameters
        -----------------------
        node_other : SearchTreeNode, the node to compare with.
        
        Returns
        -----------------------
        True if mu(node_current) < mu(node_other)
        """
        
        return self.mu > node_other.mu
    
    def __le__(self, node_other):
        """
        Description
        -----------------------
        Make the comparison <= between two SearchTree nodes based on their negative mu.
        This might seem confusing, but it is useful in order to use a heapq of SearchTreeNodes where
        the first element is the SearchTreeNode with maximum mu.
        
        Parameters
        -----------------------
        node_other : SearchTreeNode, the node to compare with.
        
        Returns
        -----------------------
        True if mu(node_current) <= mu(node_other)
        """
        
        return self.mu >= node_other.mu
    
    def __eq__(self, node_other):
        """
        Description
        -----------------------
        Make the comparison == between two SearchTree nodes based on their mu.
        
        Parameters
        -----------------------
        node_other : SearchTreeNode, the node to compare with.
        
        Returns
        -----------------------
        True if mu(node_current) == mu(node_other)
        """
        
        return self.mu == node_other.mu
    
    def __ne__(self, node_other):
        """
        Description
        -----------------------
        Make the comparison != between two SearchTree nodes based on their mu.
        
        Parameters
        -----------------------
        node_other : SearchTreeNode, the node to compare with.
        
        Returns
        -----------------------
        True if mu(node_current) != mu(node_other)
        """
        
        return self.mu != node_other.mu
    
    def __gt__(self, node_other):
        """
        Description
        -----------------------
        Make the comparison > between two SearchTree nodes based on their negative mu.
        This might seem confusing, but it is useful in order to use a heapq of SearchTreeNodes where
        the first element is the SearchTreeNode with maximum mu.
        
        Parameters
        -----------------------
        node_other : SearchTreeNode, the node to compare with.
        
        Returns
        -----------------------
        True if mu(node_current) > mu(node_other)
        """
        
        return self.mu < node_other.mu
    
    def __ge__(self, node_other):
        """
        Description
        -----------------------
        Make the comparison >= between two SearchTree nodes based on their negative mu.
        This might seem confusing, but it is useful in order to use a heapq of SearchTreeNodes where
        the first element is the SearchTreeNode with maximum mu.
        
        Parameters
        -----------------------
        node_other : SearchTreeNode, the node to compare with.
        
        Returns
        -----------------------
        True if mu(node_current) >= mu(node_other)
        """
        
        return self.mu <= node_other.mu
    
    def predict(self, X, method='mc'):
        """
        Description
        -----------------------
        Predict the class of data point X.

        Attributes & Parameters
        -----------------------
        X      : np.array, the data point to be classified.
        method : String in ['mc', 'nb'].
                  - 'mc' for majority class.
                  - 'nb' for naive bayes.
        
        Returns
        -----------------------
        Int, the predicted class of data point X.
        """
        
        return self.tree.predict(X, method)
    
    def test(self, stream_generator, method='mc', n=1000):
        """
        Description
        -----------------------
        Test the resulting DT on a number of samples and return its accuracy.

        Parameters
        -----------------------
        stream_generator : StreamGenerator object.
        method           : String in ['mc', 'nb'].
                            - 'mc' for majority class.
                            - 'nb' for naive bayes.
        n                : Int, the number of samples to use for the test.
        
        Returns
        -----------------------
        Float in [0, 1], the accuracy.
        """
        
        r = 0
        for i in range(n):
            X, y = stream_generator.generate()
            y_pred = self.predict(X, method)
            r += (y_pred == y)

        return r/n
    
class SearchTree:
    """
    Description
    -----------------------
    Class describing a Search Tree.
    """
    
    def __init__(self, root):
        """
        Description
        -----------------------
        Constructor of class SearchTree.

        Attributes & Parameters
        -----------------------
        root : SearchTreeNode, the root of the search tree. If None
        """
        
        self.root = root
            
    def select(self):
        """
        Description
        -----------------------
        Select the most promising node for TCDT.

        Parameters
        -----------------------
        
        Returns
        -----------------------
        node : SearchTreeNode, the selected SearchNode.
        """
        
        node = self.root
        node.update_visits()
        while (node.fully_expanded) and (node.children):
            mu_max = -1
            child_max = None
            for child in chain.from_iterable(node.children.values()):
                mu = child.sample()
                if mu > mu_max:
                    mu_max = mu
                    child_max = child
                    
            node = child_max
            node.update_visits()
            
        return node
        
    def simulate(self, node, stream_generator, n_samples):
        """
        Description
        -----------------------
        Simulation step of TCDT.

        Parameters
        -----------------------
        node             : SearchTreeNode, the selected SearchTree node for the simuation step.
        stream_generator : StreamGenerator object.
        n_samples        : Int, the number of samples to use in the simulation step.
        
        Returns
        -----------------------
        """
        
        for i in range(n_samples):
            X, y = stream_generator.generate()
            leaf = node.update_statistics(X, y)
            
    def expand(self, searchtree_node, thresh_leaf, thresh_tree):
#    def expand(self, searchtree_node, thresh_leaf, thresh_mu, thresh_sigma):
        """
        Description
        -----------------------
        Expand a SearchNode.

        Parameters
        -----------------------
        searchtree_node : SearchTreeNode, the selected SearchNode.
        thresh_leaf     : Float, small threshold used with a heuristic to decide whether a leaf in a Decision Tree is pure or not.
        
        Returns
        -----------------------
        """
        
        if searchtree_node.fully_expanded:
            raise Exception("The node is fully expanded")
            
        leaf = heapq.heappop(searchtree_node.leaves_active)
        return searchtree_node.expand(leaf, thresh_leaf, thresh_tree)
#        return searchtree_node.expand(leaf, thresh_leaf, thresh_mu, thresh_sigma)
            
    def backpropagate(self, searchtree_node):
        """
        Description
        -----------------------
        Backpropagation step of TSDT.

        Parameters
        -----------------------
        searchtree_node : SearchTreeNode, SearchNode to use for backpropagation.
        
        Returns
        -----------------------
        """

        current_node = searchtree_node
        parent = current_node.parent
        while parent is not None:
            mu_max, sigma_max = -1, 0
            # Gather the parameters of all normal approximations of the children's distrubtions.
            for child in chain.from_iterable(parent.children.values()):
                assert child.mu is not None, "mu should not be NaN"
                assert child.sigma is not None, "sigma should not be NaN"
#                start_time = time()
                if child.mu > mu_max:
                    mu_max, sigma_max = child.mu, child.sigma
                    
#                print(time() - start_time)
                
            # Update (mu, sigma) for the normal approximation of the prior of the parent Search Node.
            parent.replace_mu_sigma(mu_max, sigma_max)
            current_node = parent
            parent = current_node.parent
                
    def run(self, stream_generator, n_iter, n_samples=100, thresh_tree=1e-6, thresh_leaf=1e-6, thresh_mu=0.8, thresh_sigma=0.1, print_iter=10, time_limit=600):
        """
        Description
        -----------------------
        Run TCDT for a number of iterations. If we reach the stopping condition before the end of the iterations,
        then return the reached SearchTreeNode, otherwise return nothing and use method infer to infer a well performing SerarchTreeNode.

        Parameters
        -----------------------
        stream_generator : StreamGenerator object.
        n_iter           : Int, number of iterations.
        n_samples        : Int, number of samples to use in the simulation step.
        thresh_tree      : Float, small threshold on the tree value, used as a stopping criterion.
        thresh_leaf      : Float, small threshold used with a heuristic to decide whether a leaf in a Decision Tree is pure or not.
        thresh_mu        : Float, threshold of mu above which we check for the stopping criterion.
        thresh_sigma     : Float, threshold of sigma below which we check for the stopping criterion.
        print_iter       : Int, number of iterations between two consecutive prints.
        
        Returns
        -----------------------
        SearchTreeNode if the stopping criterion is reached, otherwise return nothing.
        """
        
        start_time = time()
        for i in range(n_iter):
#            start_time = time()
            searchtree_node = self.select()
#            print('Selection:', time() - start_time)
#            start_time = time()
            self.simulate(searchtree_node, stream_generator, n_samples)
#            print('Simulation:', time() - start_time)
            if not searchtree_node.activated:
                searchtree_node.activate_all()
                
            if (searchtree_node.to_expand) and (not searchtree_node.fully_expanded):
#                start_time = time()
                node = self.expand(searchtree_node, thresh_leaf, thresh_tree)
#                node = self.expand(searchtree_node, thresh_leaf, thresh_mu, thresh_sigma)
                if node is not None:
                    print('Stopping criterion reached after %d iterations, no need to infer the best performing SearchTreeNode.' %i)
                    return node
#                print('Expansion:', time() - start_time)
#                start_time = time()
                self.backpropagate(searchtree_node)
#                print('Backpropagation:', time() - start_time)
                
            else:
                searchtree_node.update_mu_sigma()
                searchtree_node.tree.update_value()
                if searchtree_node.tree.value > -thresh_tree:
#                if (searchtree_node.mu > thresh_mu) and (searchtree_node.sigma < thresh_sigma):
                    print('Stopping criterion reached after %d iterations, no need to infer the best performing SearchTreeNode.' %i)
                    return searchtree_node
                
#                start_time = time()
                self.backpropagate(searchtree_node)
#                print('Backpropagation:', time() - start_time)
                
            if i%print_iter == 0:
                print('iteration : %d' %i)
                
            if time() - start_time > time_limit:
                return
                
#            print('iteration time:', time() - start_time)
#            print('\n')
                
    def infer(self, method='n_visits', n_min=5):
        """
        Description
        -----------------------
        Infer the best SearchNode after training.

        Parameters
        -----------------------
        method : String in {'n_visits', 'max_value', 'mixed'}.
                    - n_visits  : Choose the most visited child node.
                    - max_value : Choose the child node maximising the estimated value.
                    - mixed     : Start with the method n_visits until the number of visits becomes smaller than some threshold, then continue
                                  with method max_value.
        n_min  : Int, minimum n_visits to switch from method n_visits to max_value. Only used with method mixed.
        
        Returns
        -----------------------
        SearchTreeNode, the inferred SearchNode.
        """
        
        node = self.root
        
        if method == 'n_visits':
            children = list(chain.from_iterable(node.children.values()))
            while children:
                children_visits = [child.n_visits for child in children]
                node = children[np.argmax(children_visits)]
                children = list(chain.from_iterable(node.children.values()))
                
        elif method == 'max_value':
            while node.children:
                mu_max = -1
                child_max = None
                for child in chain.from_iterable(node.children.values()):
                    mu = child.mu
                    if mu > mu_max:
                        mu_max = mu
                        child_max = child
                
                node = child_max
                
        elif method == 'mixed':
            children = list(chain.from_iterable(node.children.values()))
            children_visits = [child.n_visits for child in children]
            index_max = np.argmax(children_visits)
            while (children) and (children[index_max].n_visits >= n_min):
                node = children[index_max]
                children = list(chain.from_iterable(node.children.values()))
                children_visits = [child.n_visits for child in children]
                index_max = np.argmax(children_visits)
                
            while node.children:
                mu_max = -1
                child_max = None
                for child in chain.from_iterable(node.children.values()):
                    mu = child.mu
                    if mu > mu_max:
                        mu_max = mu
                        child_max = child
                
                node = child_max

        return node
        
def approximate_beta(alpha, beta, gamma=0.5):
    """
    Description
    -----------------------
    Normal approximation of a Beta distribution.

    Parameters
    -----------------------
    alpha, beta : Floats in [0, 1], parameters of the Beta distribution to approximate.
    gamma       : Float > 0 , the exponent used to scale the std of the posterior distribution.

    Returns
    -----------------------
    mu, sigma : Floats, mean and std of the Normal approximation of the Beta distribution
    """
    
#    return alpha/(alpha + beta), (np.sqrt(alpha*beta/((alpha+beta)**2*(alpha+beta+1))))**gamma
    return alpha/(alpha + beta), np.sqrt(alpha*beta/((alpha+beta)**2*(alpha+beta+1)))

def approximate_max_normal_2(params):
    """
    Description
    -----------------------
    Normal approximation of the maximum of two independent Normal distributions.

    Parameters
    -----------------------
    params : List of two tuples, each tuple describes the mean and std of a Normal distribution.

    Returns
    -----------------------
    mu, sigma : Floats, mean and std of the Normal approximation of the maximum.
    """
    
    mu1, sigma1 = params[0]
    mu2, sigma2 = params[1]
    sigma_m = sigma1**2 + sigma2**2
    alpha = (mu1 - mu2)/sigma_m
    PHI_alpha, PHI_alpha_ = norm.cdf(alpha), norm.cdf(-alpha)
    phi_alpha = norm.pdf(alpha)
    mu = mu1*PHI_alpha + mu2*PHI_alpha_ + sigma_m*phi_alpha
    sigma = np.sqrt((mu1**2 + sigma1**2)*PHI_alpha + (mu2**2 + sigma2**2)*PHI_alpha_ + (mu1 + mu2)*sigma_m*phi_alpha - mu**2)
    return mu, sigma

def approximate_max_normal(params):
    """
    Description
    -----------------------
    Normal approximation of the maximum of independent Normal distributions.

    Parameters
    -----------------------
    params : List, each element is a tuple with the mean and std of a Normal distribution.

    Returns
    -----------------------
    mu, sigma : Floats, mean and std of the Normal approximation of the maximum.
    """
    
    assert params, "params should not be empty"
    
    params_len = len(params)
    if params_len == 1:
        return params[0]
    
    elif params_len == 2:
        return approximate_max_normal_2(params)
        
    else:
        normal1 = params.pop()
        return approximate_max_normal_2([normal1, approximate_max_normal(params)])
