"""
This module implements decision tree classifier used to cluster
FCS event in cluster of same sizes (approximatively same size 
in case of histograms).

"""
import numpy as np 
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, ClusterMixin
from itertools import repeat

class HistogramTransform(BaseEstimator, TransformerMixin):
    """Apply an histogram transform to the data.

    """

    def __init__(self, edges):
        """
        Parameters:
        -----------
        edges : dct, shape (n_channels, )
                Dictionary with key:value as follow
                'channel_id':edges with edges an array 
                containing the edges of the bins along a 
                particular channel.

        """
        self.edges = edges

    def fit(self, X, y=None):
        """No parameters to estimate.

        """
        return self

    def transform(self, X, y=None):
        """Create an histogram according to the bins.
        Parameters:
        -----------
        X : FCMeasurement,
            Contains the flow cytometer data.

        """
        #extract the volume from the meta data
        V = float(X.get_meta()['$VOL'])

        #extract only the colums of interest
        X_ = X[list(self.edges.keys())]
        sorted_keys = [c for c in list(X_.columns) if c in list(self.edges.keys())]

        #construct the multidimensional histogram
        H, edges = np.histogramdd(X_.values, bins=[self.edges[key] for key in sorted_keys])
        edges = np.array(edges)

        #get the bins centers
        centers = edges[:, 0:-1] + (edges[:, 1] - edges[:, 0]).reshape(-1, 1) / 2

        hist = pd.DataFrame(columns=list(self.edges.keys()) + ['counts'])
        bin_sizes = [len(e) for e in centers]

        nb_copies = np.cumprod([1] + bin_sizes[0:-1])
        nb_repeat = np.cumprod([1] + bin_sizes[-1:0:-1])[::-1]

        for (c, name) in enumerate(X_.columns):
            hist[name] = np.array(list(map(lambda e: [e] * nb_repeat[c], centers[c])) * nb_copies[c]).flatten()

        hist['counts'] = np.array(H).flatten() * (9E+4 / V)
                 
        return hist


class DTClassifier(BaseEstimator, TransformerMixin, ClusterMixin):
    """Cluster a dataset in clusters with same cardinality by recursively
    splitting the dataset along the axis of maximal variance. The splits are
    done using the median value so that each split has the same number of 
    sample.

    """

    def __init__(self, max_depth=3, columns=None, weight_decay=None):
        """
        Parameters:
        -----------
        max_depth : int, defaults to 3.
                    Maximal depth of the recurence
        
        columns : list, defaults to None.
                  Apply the clustering along the columns specified only.

        weight_decay : float, [0; 1], default to None.
                       If None the decision tree classifier is fit on the
                       data X.

                       If not None the DC_classifier must follow a
                       Histogram_transform in the pipeline and the decision
                       tree classifier is fit on the exponentially weighted
                       moving average (ewma). Note that this require that the 
                       ewma is initialized before calling fit on the pipeline.
                       Larger weight decay discard contribution of old FCS
                       faster and a weight decay of zero corresponds to a 
                       constant mean histogram fixed to the initialized values.

        
        """
        self.max_depth = max_depth
        self.columns = columns
        self.weight_decay = weight_decay

    def fit(self, X, y=None):
        """Build the decision tree.

        Parameters:
        -----------
        X : pandas DataFrame, shape (n_events, n_channels)
            List of (n_channels)-dimensional data points. Each row
            corresponds to a single event.

        Returns:
        --------
        self : this estimator (to be compatible with sklearn API).

        """
        def recursive_fit(x, depth=0, branch=None, queue=[], tree=[]):
            """Recursive clustering of the data.

            Parameters:
            -----------
            x : pandas DataFrame, shape (n_events, n_channels)
                Input data at current node.

            depth : int
                    depth of the node from which the current branch is 
                    leaving.
            
            branch : list, shape (3,)
                     branch leading to the current node in the decision 
                     tree. (splitting_variable, median, result) with 
                     'splitting_variable' the column with maximal variance,
                     'median' the value of the median along this column, 
                     'result' the result of the >= operator.
            
            queue : list, shape (depth, )
                    concatenation of all the branches leading to the 
                    current state.

            tree : pandas DataFrame, shape (2^max_depth, max_depth)
                   list of all the branches from initial node to leaf node.
            
            Returns:
            tree : see above.

            """

            if branch:
                queue.append(branch)

            if depth < self.max_depth:
                #compute the branch varible
                if 'counts' in list(x.columns):
                    means = x[self.columns].mean(axis=0)
                    variances = np.square((x[self.columns] - means)).multiply(x['counts'], axis=0).sum()
                    splitting_variable = variances.idxmax()
                else:
                    splitting_variable = x[self.columns].var(axis=0).idxmax()

                if 'counts' in list(x.columns):
                    cumsum = x[[splitting_variable, 'counts']].groupby(by=splitting_variable, sort=False).sum().cumsum()
                    median = (cumsum >= cumsum.iloc[-1]/2).idxmax()[0]
                else:
                    median = x[splitting_variable].median(axis=0)

                mask = (x[splitting_variable] > median).values
                #handle the case where the values are equal to the mdian (e.g. projection on one axis)
                idx_switch = np.random.permutation(np.where((x[splitting_variable] == median)))
                idx_switch = idx_switch[:, :np.max([0, int(np.floor(0.5 * mask.size - np.sum(mask)))])].squeeze()
                mask[idx_switch] = np.logical_not(mask[idx_switch])

                #recursion
                recursive_fit(x.loc[mask, :], 
                              depth+1, 
                              (splitting_variable, median, True),
                              queue.copy(), 
                              tree)

                recursive_fit(x.loc[np.logical_not(mask), :],
                              depth+1,
                              (splitting_variable, median, False),
                              queue.copy(),
                              tree)
            else:
                #stopping condition
                tree.append(queue)
                return tree
            
            return tree

        Xt = X

        if self.weight_decay is not None:
            #fit the decision tree on the ewma of the previous step
            Xt = self.ewma
            if self.weight_decay > 0:
                #update the ewma with the new histogram
                self.ewma.loc[:, 'counts'] = self.weight_decay * X['counts'].values + (1 - self.weight_decay) * self.ewma['counts'].values

        if self.columns:
            self.tree_ = pd.DataFrame(recursive_fit(Xt[self.columns + ['counts'] if 'counts' in list(Xt.columns) else self.columns]))
        else:
            self.tree_ = pd.DataFrame(recursive_fit(Xt))
            
        return self


    def predict(self, X, y=None):
        """Cluster the data using the fitted decision tree.

        Parameters:
        -----------
        X : pandas DataFrame, shape (n_events, n_channels)
            list of (n_channels)-dimensional data points. Each row
            corresponds to a single event.

        Returns:
        --------
        labels_ : pandas DataFrame containing the cluster index for
                  each event.
        """

        def recursive_predict(x=X, tree=self.tree_.copy(), label_cursor=1):
            """Recursive clustering of the data.

            Parameters:
            -----------
            X : pandas DataFrame, shape (n_events, n_channels)
                list of (n_channels)-dimensional data points. Each row
                corresponds to a single event.

            tree : pandas DataFrame, shape (2^max_depth, max_depth)
                   list of all the branches from initial node to leaf node.

            label_cursor : counter giving the current cluster index.
            
            Returns:
            --------
            labels_cursor : see above.

            """

            if tree.shape[1]:
                #get the 2 truncated trees (trees after the 2
                #branches leaving the current node)
                grp = tree.groupby(by=list(tree.columns)[0], sort=False)
                branches = list(grp.groups.keys())

                mask = ((x[branches[0][0]] > branches[0][1]) == branches[0][2]).values
                #handle the case where the value is equal to the median (e.g. projection)
                idx_switch = np.random.permutation(np.where((x[branches[0][0]] == branches[0][1])))
                idx_switch = idx_switch[:, :np.max([0, int(np.floor(0.5 * mask.size - np.sum(mask)))])].squeeze()
                mask[idx_switch] = np.logical_not(mask[idx_switch])
            
                #recursion
                label_cursor = recursive_predict(x.loc[mask, :], 
                                                 grp.get_group(branches[0]).drop(list(tree.columns)[0], axis=1), 
                                                 label_cursor)

                label_cursor = recursive_predict(x.loc[np.logical_not(mask), :], 
                                                 grp.get_group(branches[1]).drop(list(tree.columns)[0], axis=1), 
                                                 label_cursor)

            else:
                X.loc[x.index, 'cluster_ID'] = label_cursor
                label_cursor += 1
                return label_cursor
            
            return label_cursor
        
        X['cluster_ID'] = 0
        recursive_predict()
        self.labels_ =  X['cluster_ID'].values
        
        return self.labels_

    
    def transform(self, X, y=None):
        """Given a dataset return the count per bin.

        Parameters:
        -----------
        X : pandas DataFrame, shape (n_events, n_channels)
            list of (n_channels)-dimensional data points. Each row
            corresponds to a single event.

        Returns:
        --------
        Number of counts per bin.

        """
        self.predict(X)

        if 'counts' in list(X.columns):
            df = pd.DataFrame({'counts':X['counts'], 'labels':self.labels_})
            return np.atleast_1d(df.groupby(by='labels').sum().values.squeeze())
        else:
            return np.histogram(self.labels_, len(np.unique(self.labels_)))[0]
    
    def initialize_ewma(self, fcms, preprocessing, edges):
        """Initialize the exponentialy weighted moving average histogram
        with the mean over multiple FCM.

        Parameters:
        -----------
            fcms : iterable,
                   Iterable pointing toward FCM files.

            preprocessing : FCTFunction,
                            Lambda function applying the 
                            FLowCytometryTools preprocessing transform
                            and gating to the FCM.

            edges : dct, shape (n_channels, )
                    Dictionary with key:value as follow
                    'channel_id':edges with edges an array 
                    containing the edges of the bins along a 
                    particular channel.
        """

        #instanciate the histogram transformer
        hist = HistogramTransform(edges)
        N = len(fcms)

        #initialize the mean histogram
        self.ewma = hist.transform(fcms[0])

        for i in np.arange(1, len(fcms)):
            self.ewma['counts'] += hist.transform(preprocessing.transform(fcms[i]))['counts']
        
        self.ewma['counts'] /= N