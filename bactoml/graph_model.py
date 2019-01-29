"""
This module implements the graph model for outlier detection.

Each FCS file corresponds to a node of the graph and has features
attributed: 
 - its features ('coordinates in space'), the features extracted by the pipeline
 - the mean distance between itself and the rest of the nodes
 - its closeness centrality in the graph
 - its label, -1 for outlier and 1 for inliner

When a node is added to the network the mean distance of all the nodes is
updated and the average mean distance for the whole network is computed.
An edge is formed between the new node and its neighbor which are closer
than the average mean distance in the network. The closeness centrality 
of the new node is computed and if it falls bellow a fixed thershold the
point is labeled as an outlier.

"""

import numpy as np
import pandas as pd
import networkx as nx

import re
import itertools

from sklearn.base import BaseEstimator, TransformerMixin    
from scipy.spatial import distance
from functools import partial

class GraphModel(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns, cc_threshold=0.2, distance_func=distance.euclidean):
        """
        Parameters:
        -----------
        columns : list of string or regular expression
                  selects the columns / features used for the distance measurement
        cc_threshold : float,
                       Closeness centrality threshold to determine outlier
        distance_func : function,
                        Function that takes two samples and returns a 
                        distance measure. 

        """
        self.columns = columns
        self.cc_threshold = 0.2
        self.distance_func = distance_func

    def fit(self, X, y=None):
        """
        Parameters:
        -----------
        X : pandas DataFrame,
            Contains the features extracted from one FCMeasurement.

        Returns:
        --------
        self : GraphModel,
               This instance.

        """
        return self

    def initialize_graph(self, X, pp_pipe):
        """Initialize the graph model.
        Initialize the graph and coordinate and outlier queues.
        
        Parameters:
        -----------
        X : pandas DataFrame,
            Contains the features from the samples used to initialize
            the graph model.
        pp_pipe : pipeline,
                  Pre-processing pipeline used to generate the features
                  from FCMeasurement instances.
        
        """

        # set the name of the attributes used for distance measurement
        if isinstance(self.columns, re.Pattern):
            self.col = list(filter(lambda s : self.columns.match(s), X.columns))
        else:
            self.col = self.columns


        def get_mean_distance(row, df):
            """Compute the mean distance between one row and all the row
            on the dataset.

            Parameters:
            -----------
            row : pandas DataFrame,
                  Row of a pandas dataframe representing the current sample.
            df : pandas DataFrame,
                 Contains all the sample for which the distance to the current
                 sample (row) is computed.

            Returns:
            --------
            mean_distance : float,
                            mean distance between the current sample and all the
                            samples in the dataset.

            """
            mean_distance = 0
            
            for _, vals in df.iterrows():
                mean_distance += self.distance_func(vals.values, row.values)
            mean_distance /= len(list(df.iterrows())) - 1
            
            return mean_distance
        
        # features and mean_distance for all the points in the graph
        features = pp_pipe.fit_transform(X)[self.col]
        mean_distance = features.apply(partial(get_mean_distance, df=features), axis=1)
        features = features.values

        # intialize the graph
        self.graph = nx.Graph(dist_threshold=mean_distance.mean())

        # add the nodes with attributes to the graph
        keys = ['features', 'mean_distance']
        nodes_attributes_dct = list(zip(range(len(features)), map(lambda x: dict(zip(keys, x)), zip(features, mean_distance))))

        self.graph.add_nodes_from(nodes_attributes_dct)

        # update the graph edges
        edges = filter(lambda x: self.distance_func(x[0][1]['features'], x[1][1]['features']) < self.graph.graph['dist_threshold'], itertools.combinations(self.graph.nodes(data=True), 2))
        self.graph.add_edges_from(map(lambda x: (x[0][0], x[1][0]), edges))

        # update the nodes attributes
        nx.set_node_attributes(self.graph, dict(zip(range(len(features)), nx.closeness_centrality(self.graph))), 'closeness_centrality')
        nx.set_node_attributes(self.graph, {key : (val < 0.2) for key, val in nx.get_node_attributes(self.graph, 'closeness_centrality').items()}, 'labels')


    def transform(self, X, y=None):
        """
        Parameters:
        -----------
        X : pandas DataFrame,
            Contains the features extracted from one FCMeasurement.

        Returns:
        --------
        out : pandas DataFrame,
              Contains -1 for outlier, 1 for inliner.

        """
        # add new node with 'features' attributes
        length_i = self.graph.number_of_nodes()
        length_f = length_i + 1
        new_idx = self.graph.number_of_nodes()
        
        self.graph.add_node(new_idx, features=X[self.col].values, mean_distance=0.0)

        # compute the 'features' distance between the new and precedent nodes and update the 'mean_distance'
        distances = []

        for node, idx in zip(self.graph.nodes(data=True), range(length_i)):
            dist = self.distance_func(self.graph.nodes(data=True)[new_idx]['features'], node[1]['features'])
            distances.append(dist)
            self.graph.node[new_idx]['mean_distance'] += dist
            self.graph.node[idx]['mean_distance'] = (length_i * self.graph.node[idx]['mean_distance'] + dist) / length_f

        self.graph.node[new_idx]['mean_distance'] /= length_i

        # update the 'dist_threshold' graph attribute
        self.graph.graph['dist_threshold'] = np.mean(list(nx.get_node_attributes(self.graph, 'mean_distance').values()))

        # update the graph edges
        edges = filter(lambda x: x[1] < self.graph.graph['dist_threshold'], zip(range(length_i), distances))
        self.graph.add_edges_from(map(lambda x: (new_idx, x[0]), edges))

        #compute the closeness centrality between the new and precedent nodes
        closeness_centrality = nx.closeness_centrality(self.graph, u=new_idx)

        self.graph.node[new_idx]['closeness_centrality'] = closeness_centrality
        self.graph.node[new_idx]['labels'] = closeness_centrality < 0.2

        return pd.DataFrame(data=[self.graph.node[new_idx]['closeness_centrality']], columns=['labels'])