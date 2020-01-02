import numpy as np
import pandas as pd

from scipy.spatial import distance
from sklearn.base import BaseEstimator, TransformerMixin  
from sklearn.neighbors import NearestNeighbors

class NNDistanceModel(BaseEstimator, TransformerMixin):

    """
    Nearest neighbor distance model which calculates the outlier score. 
    The outlier score is the distance to the nearest neighbour.
    """
    
    def __init__(self, columns, distance_func=distance.euclidean, n_neighbors = 1):
        
        """

        Parameters:
        -----------
        columns : list of string or regular expression
                  selects the columns / features used for the distance measurement
        distance_func: function,
                        Function that takes two samples and returns a 
                        distance measure. 
        n_neighbors: number of neighbors to be used in the scikit-learn 
                    NearestNeighbors model
        """
        self.columns = columns
        self.distance_func = distance_func
        self.kN = n_neighbors

    def fit(self, X, y=None):

        """Fits the model to a dataset

        Parameters:
        -----------

        X : pandas DataFrame,
            Contains the features extracted from one FCMeasurement.

        Returns:
        --------

        self : this estimator (to be compatible with sklearn API).


        """

        self.neigh = NearestNeighbors(self.kN, metric = self.distance_func).fit(X[self.columns])

        return self

    def transform(self, X, y=None):

        """Apply the NN distance model for outlier detection to the 
        preprocessed dataset and return for each measurement the outlier score

        Parameters:
        -----------

        X : pandas DataFrame,
            Contains the features extracted from one FCMeasurement.

        Returns:
        --------
        
        output : pandas DataFrame with the outlier score

        """

        output = []
        for i, row in X[self.columns].iterrows():
            distance_, _ = self.neigh.kneighbors([row], self.kN, return_distance = True)
            output.append(distance_[0][0])

        return pd.DataFrame(output, columns=['outlier_score'], index = X.index)