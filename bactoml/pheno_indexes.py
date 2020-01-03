import numpy as np
import pandas as pd
import itertools

from sklearn.base import BaseEstimator, TransformerMixin  
from sklearn.neighbors.kde import KernelDensity

class PhenoIndexes(BaseEstimator, TransformerMixin):
    """
    Calcultes the Phenotypic Diversity indexes. """

    def __init__(self, edges_pheno, kernel = 'gaussian', d = 4, bw = 0.01, downsampling = False, downsample_perc = 0.5):

        """ 
        Parameters:
        -----------

        edges_pheno: dictionary with the values on which the density is sampled in each channel.
        columns: columns on which the indexes are calculated. They have to be the same than the keys in the edges dictionary. 
        d: decimal rounding factor. Default value = 4.
        bw: bandwidth of the fitted Kernel density. 
        downsampling: boolean. Determines if downsampling is performed or not.
        downsample_perc: float in the interval [0, 1]. Percentage of the datapoints to be retained."""

        self.edges_pheno = edges_pheno
        self.columns = list(self.edges_pheno.keys())
        self.kernel = kernel
        self.d = d
        self.bw = bw
        self.downsampling = downsampling
        self.downsample_perc = downsample_perc

        if self.downsampling:
            if (self.downsample_perc > 1) | (self.downsample_perc <= 0):
                raise ValueError('The downsampling percentage has to be between 0 and 1')

    def fit(self, X, y=None):
        """No parameters to estimate.

        Parameters:
        -----------

        X : pandas DataFrame,
            Contains the features extracted from one FCMeasurement.

        Returns:
        --------

        self :  This instance.

        """
        return self

    def transform(self, X, y = None):

        """
        Transforms the FCMeasurements into Phenotypic Diversity indexes. 

        Parameters:
        -----------

        X : pandas DataFrame, shape (n_events, n_channels)
            list of (n_channels)-dimensional data points. Each row
            corresponds to a single event. 
            
         Returns:
        --------

        output : numpy array with phenotypic diversity indexes D0, D1 and D2

        """   
        

        # diversity formulas
        d0 = lambda x: len(x)
        d1 = lambda x: np.exp(-sum((x)*np.log(x)))
        d2 = lambda x: 1/sum((x)**2)

        if self.downsampling:
            np.random.seed(seed=42)
            X = X[self.columns].iloc[np.random.choice(X.shape[0], int(X.shape[0]*self.downsample_perc), replace = False)]


        ### calculate the "fingerprint"
        binned_density = []

        for cp in itertools.combinations(self.columns, 2):
            channel_pair = list(cp)
            kde = KernelDensity(kernel = self.kernel, bandwidth = self.bw, rtol = 1E-4).fit(X[channel_pair])
            Xt, Yt = np.meshgrid(self.edges_pheno[channel_pair[0]], self.edges_pheno[channel_pair[1]])
            Z = kde.score_samples(np.column_stack((Xt.flatten(),Yt.flatten())))
            binned_density.extend(Z)

        ### calculate the phenotypic diversity index

        # take the exponential (because kde.score_samples returns logarithmic values) and normalize
        dens = np.exp(np.array(binned_density))
        dens = dens/max(dens)
        # only retain the values up to a rounding factor d
        dens = np.round(dens, self.d)
        dens = dens[dens!=0]
        
        if sum(dens) == 0:
            div0 = 0
            div1 = 0
            div2 = 0        
        else:
            dens = dens/sum(dens)
            div0 = d0(dens)
            div1 = d1(dens)
            div2 = d2(dens)
            
        return np.array([div0, div1, div2])
