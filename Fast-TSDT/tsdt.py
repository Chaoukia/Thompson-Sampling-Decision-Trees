from searchtree import *

class TSDT:
    """
    Description
    -----------------------
    Class describing the TCDT algorithm.
    """
    
    def __init__(self, categories, K, metric='gini', min_samples=2):
        """
        Description
        -----------------------
        Constructor of class TCDT.

        Attributes & Parameters
        -----------------------
        categories  : np.array of shape (n_attributes,), the number of categories of each attribute.
        K           : Int, the number of classes.
        metric      : String in ['entropy', 'gini', 'majority'].
        min_samples : Int, the minimum number of samples for a decision tree leaf to be considered active.
        """
        
        self.categories = categories
        self.K = K
        self.metric = metric
        self.min_samples = min_samples
        
    def run(self, stream_generator, n_iter, n_samples, thresh_tree, thresh_leaf, thresh_mu, thresh_sigma, method, lambd, gamma, print_iter, time_limit):
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
        gamma            : Float > 0 , the exponent used to scale the std of the posterior distribution.
        method           : String in {'n_visits', 'max_value', 'mixed'}.
                              - n_visits  : Choose the most visited child node.
                              - max_value : Choose the child node maximising the estimated value.
                              - mixed     : Start with the method n_visits until the number of visits becomes smaller than some threshold, then continue
                                            with method max_value.
        print_iter       : Int, number of iterations between two prints.
        time_limit       : Int, time limit in seconds.
        
        Returns
        -----------------------
        node             : SearchTreeNode, the proposed solution.
        searchtree.      : SearchTree, the Search Tree, useful for analysis purposes.
        """
        
        tree = DecisionTree(self.categories, self.K, min_samples=self.min_samples, metric=self.metric)
        searchtree_node = SearchTreeNode(tree, 0, min_samples=self.min_samples, lambd=lambd, gamma=gamma)
        searchtree = SearchTree(searchtree_node)
        searchtree.simulate(searchtree.root, stream_generator, n_samples)
        searchtree.root.update_mu_sigma()
        node = searchtree.run(stream_generator, n_iter, n_samples, thresh_tree, thresh_leaf, thresh_mu, thresh_sigma, print_iter, time_limit)
        if node is None:
            node = searchtree.infer(method)
            
        return node, searchtree