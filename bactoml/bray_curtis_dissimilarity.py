"""
This module implements the Bray-Curtis dissimilarity for outlier detection,
as described by Favere et al. 2020. 
The model is fitted on the fingerprints of a reference set representative for
normal operating conditions. In a transform step the Bray-Curtis dissimilarity is
calculated which is the average of the Bray-Curtis dissimilarity of a given sample
to all the samples in the reference set.

"""


import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin  

def BrayCurtisDissimilarityDistance(x, y):
    return 1-2*sum(np.minimum(x,y))/(sum(x)+sum(y))

class BrayCurtisDissimilarity(BaseEstimator, TransformerMixin):

    """
    Bray-Curtis Dissimilarity model described by Favere et al. 2020.
    """
    
    def __init__(self, columns):
        
        """

        Parameters:
        -----------
        columns : list of string or regular expression
                  selects the columns / features used for the distance measurement
        """
        self.columns = columns

    def fit(self, X, y=None):

        """Fits the model to a dataset

        Parameters:
        -----------

        X : pandas DataFrame,
            Contains the features extracted from one FCMeasurement.

        Returns:
        --------

        self 
                this estimator (to be compatible with sklearn API).


        """

        self.X_ref = X[self.columns]

        return self

    def transform(self, X, y=None):

        """Apply the model to the online fingerprints and return 
        the Bray-Curtis dissimilarities

        Parameters:
        -----------

        X : pandas DataFrame,
            Contains the features extracted from one FCMeasurement.

        Returns:
        --------
        
        output 
                pandas DataFrame with the dissimilarities

        """

        output = []
        for _, row in X[self.columns].iterrows():
            dissimilarities = self.X_ref.apply(lambda x: BrayCurtisDissimilarityDistance(x, row), axis = 1)
            output.append(np.average(dissimilarities))

        return pd.DataFrame(output, columns=['BC_dissimilarity'], index = X.index)