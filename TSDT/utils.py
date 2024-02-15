import numpy as np
import re


iteration = lambda s : int(re.split('_', s[:-4])[-1])

class StreamGenerator:
    """
    Description
    -----------------------
    Class describing a synthetic stream generator.
    """
    
    def __init__(self, generator):
        """
        Description
        -----------------------
        Constructor of class StreamGenerator.

        Attributes & Parameters
        -----------------------
        generator    : Dict, - keys   : Attributes.
                             - values : List of probability masses of each category of the corresponding feature.
        categories   : List, the number of categories of each attribute.
        n_attributes : Int, the number of attributes.
        """
        
        self.generator = generator
        self.categories = [len(self.generator[attribute]) for attribute in self.generator]
        self.n_attributes = len(self.categories)
        
    def generate(self):
        """
        Description
        --------------
        Generate a data point and its label.
        
        Parameters
        --------------
        
        Returns
        --------------
        List, the generated sample.
        Int, the corresponding label.
        """
            
        data_point = [np.random.choice(self.categories[i], p=self.generator[i]) for i in range(self.n_attributes)]
        return data_point, self.concept(data_point)
        
    def concept(self, data_point):
        """
        Description
        --------------
        Define the concept labeling the data points.
        
        Parameters
        --------------
        data_point : List, the data point to label.
        
        Returns
        --------------
        Int in {0, 1}, the label of the data point.
        """
        
        if ((data_point[0] == 0) and (data_point[1] == 0)) or ((data_point[0] == 1) and (data_point[1] == 1)):
            return 0
        
        else:
            return 1
        
        
class StreamGeneratorReal:
    """
    Description
    -----------------------
    Class describing a stream generator of real world data.
    """
    
    def __init__(self, data):
        """
        Description
        -----------------------
        Constructor of class StreamGeneratorReal.

        Attributes & Parameters
        -----------------------
        data       : 2D np.array, the data matrix.
        categories : List, the number of categories of each attribute.
        maps       : Dict: -keys  : Int of each category.
                           -value : Dict : -keys   : value of a category.
                                           -values : Int, ineger encoding of the value.
        index      : Int, index of the current data point in the data matrix.
        """
        
        self.data = data
        self.categories = [len(set(data[:, j])) for j in range(data.shape[1]-1)]
        self.maps = self.build_maps()
        self.index = 0
        
    def build_maps(self):
        """
        Description
        --------------
        Build the maps dictionnary mapping the categorical classes of each feature to numerical values.
        
        Parameters
        --------------
        
        Returns
        --------------
        maps : Dict: -keys  : Int of each category.
                     -value : Dict : -keys   : value of a category.
                                     -values : Int, ineger encoding of the value.
        """
        
        maps = {}
        for j in range(self.data.shape[1]):
            categories = list(set(self.data[:, j]))
            categories.sort()
            maps_j = {}
            for i, category in enumerate(categories):
                maps_j[category] = i

            maps[j] = maps_j
            
        return maps
    
    def preprocess(self, u):
        """
        Description
        --------------
        Preprocess an instance of the data matrix by converting it to a numerical variable.
        
        Parameters
        --------------
        u : 1D np.array, the instance to preprocess.
        
        Returns
        --------------
        x : 1D np.array, the converted instance.
        y : Int, the corresponding label in the data matrix.
        """
        x = []
        for j, category in enumerate(u[:-1]):
            x.append(self.maps[j][category])

        y = self.maps[len(u)-1][u[-1]]
        return x, y
    
    def generate(self):
        """
        Description
        --------------
        Generate a data point and its label and move the index to the next instance in the data matrix.
        
        Parameters
        --------------
        
        Returns
        --------------
        x : 1D np.array, the converted current instance.
        y : Int, the corresponding label in the data matrix.
        """
        
        x, y = self.preprocess(self.data[self.index, :])
        self.index += 1
        if self.index == self.data.shape[0]:
            self.index = 0
            
        return x, y
    

def results(node, stream_generator, method='mc', n_samples=1000):
    """
    Description
    --------------
    Print the following results regarding a SearchNode:
        - Number of visits.
        - Number of observed samples.
        - SearchNode value.
        - Tree value.
        - Accuracy.

    Parameters
    --------------
    node             : SearchTreeNode.
    stream_generator : StreamGenerator object.
    method           : String in ['mc', 'nb'].
                        - 'mc' for majority class.
                        - 'nb' for naive bayes.
    n_samples        : Int, the number of samples to use for the test.

    Returns
    --------------
    """
    
    print('Number of visits           : %d' %node.n_visits)
    print('Number of observed samples : %d' %node.tree.n_samples)
    print('SearchNode value           : %.3f' %node.value)
    print('Tree value                 : %.3f' %node.tree.value)
    print('Accuracy                   : %.3f' %node.test(stream_generator, method, n=n_samples))
    
    