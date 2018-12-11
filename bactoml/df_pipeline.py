"""
This module implements the classes needed to integrate sk-learn Pipeline and
FeatureUnion with pandas DataFrame and FCMeasurment instances (see 
FlowCytometryTools library).

"""
import numpy as np
import pandas as pd

from types import LambdaType
from itertools import product, chain
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, _transform_one
from sklearn.externals.joblib import Parallel, delayed
from FlowCytometryTools import FCMeasurement

class DFLambdaFunction(BaseEstimator, TransformerMixin):
    """Apply a lambda function to a pandas DataFrame. 
    The implementation is compatible with the sk-learn API.

    """

    def __init__(self, func, copy=False):
        """
        Parameters:
        -----------
        func : lambda function,
               Takes a single pandas DataFrame instance as input.
        copy : boolean,
               Determine if the transform returns a copy of
               the pandas DataFrame instance or the instance itself.

        """
        self.func = func
        self.copy = copy

    def fit(self, X, y=None):
        """
        Parameters:
        -----------
        X : pandas DataFrame instance.
            Input data.

        Returns:
        --------
        self : DFLambdaFunction.
               This estimator.

        """
        if isinstance(self.func, LambdaType):
            return self

        elif isinstance(self.func, BaseEstimator):
            self.func.fit(X, y)
            return self

    def transform(self, X, y=None):
        """
        Parameters:
        -----------
        X : pandas DataFrame instance.
            Input data.

        Returns:
        --------
        pandas Dataframe instance.
        Result of the call of self.func on the pandas 
        DataFrame instance X.

        """
        X_ = X if not self.copy else X.copy()

        if isinstance(self.func, LambdaType):
            try:
                #check for iterable input
                iter(X_)
            except TypeError:
                return self.func(X_)    
            else:
                if isinstance(X_, FCMeasurement):
                    return self.func(X_) 
                else:
                    out = []
                    for i in range(len(X_)):
                        out.append(self.func(X_[i]))
                    return out
        
        elif isinstance(self.func, BaseEstimator):
            X_[X_.columns] = self.func.transform(X_.values)
            return X_

class DFInPlaceLambda(BaseEstimator, TransformerMixin):
    """Perform in place modification on the DataFrame 
    columns. 

    """

    def __init__(self, func, columns=None):
        """
        Parameters:
        -----------
        func : lambda function,
               Takes two inputs, the pandas DataFrame
               and the colums:
               DFInPlaceLambda(['TCC'], lambda C, DF : C / DF['VOL'])
        columns : array of strings,
                  Contains the name of the columns to 
                  which the function func will be 
                  applied.

        """
        self.func = func
        self.columns = columns

    def fit(self, X, y=None):
        """
        Parameters:
        -----------
        X : pandas DataFrame instance.
            Input data.

        Returns:
        --------
        self : DFLambdaFunction.
               This estimator.

        """
        return self

    def transform(self, X, y=None):
        """
        Parameters:
        -----------
        X : pandas DataFrame instance.
            Input data.

        Returns:
        --------
        pandas Dataframe instance.
        Result of the call of self.func on the columns of 
        the pandas DataFrame instance X.

        """
        X_ = X.copy()

        if self.columns:
            for c in self.columns:
                X_[c] = self.func(X_[c], X_)
        else:
            for c in X.columns:
                X_[c] = self.func(X_[c], X_)
        return X_

class DFFeatureUnion(FeatureUnion):
    """Feature union that support dataframe as inputs and
    outputs.
    Inputs can be FCMeasurement but to be able to concatenate
    the results, the output will be dataframes.

    Note : nesting the DFFeatureUnion doesn't conserve the
           columns name of the deeper DFFeatureUnion.

    """
    
    def transform(self, X):
        """Transform X separately by each transformer or
        Pipeline then concatenate the results.

        Parameters:
        -----------
        X : FCMeasurment or pandas DataFrame.
            Input data to be transformed.

        Returns:
        --------
        X_t : pandas DataFrame, shape(n_samples, 
              sum_n_components)
              hstack of results of transformers.
              sum_n_components is the sum of n_components
              (output dimension) over transformers.

        """
        Xs = Parallel(n_jobs=self.n_jobs)(
             delayed(_transform_one)(trans, weight, X)
             for name, trans, weight in self._iter())

        #get the name of the branches of the feature union and the dimension of the results
        names = list(zip(*self._iter()))[0]
        dim = list(map(lambda X : np.atleast_1d(X).size, Xs)) #if not isinstance(X, int) else 1

        #generate the name of the columns
        columns = [list(map(lambda X : '{}_{}'.format(*X), product([n], range(d)))) if d > 1 else [n] for n, d in zip(names, dim)]
        columns = list(chain.from_iterable(columns)) #flatten the array

        #flatten the list of returns
        values = np.concatenate(list(map(lambda X : X.values.flatten() if isinstance(X, pd.DataFrame) else np.atleast_1d(X), Xs)))

        X_t = pd.DataFrame(data={col : [val] for col, val in zip(columns, values)})

        return X_t

    def fit_transform(self, X, y=None, **fit_params):
        """Fit all transformers, transform the data and 
        concatenate the results.

        Parameters:
        -----------
        X : FCMeasurement or pandas DataFrame.
            Input data to be transformed.

        y : array-like, shape(n_sample, ...) optional
            Targets for supervised learning.

        Returns:
        --------
        X_t : pandas DataFrame, shape (n_samples,
              sum_n_components)
              hstack of results of transformers.
              sum_n_components is the sum of n_components
              (output dimension) over transformers.

        """
        return self.fit(X, y, **fit_params).transform(X)

class SampleWisePipeline(Pipeline):
    """Apply the whole pipeline to each sample sequentially.

    At each steps Sklearn Pipeline apply the fit/transform
    function to the whole dataset. This object apply all the
    steps to each sample before moving to the next one. This
    is usefull when dealing with prepreocessing steps, when
    the sample have different dimensions and the pipeline 
    implements a dimensionality reduction / feature selection,
    or when dealing with time series and the order of the 
    sample is important.

    """

    def __init__(self, steps, memory=None):
        """
        Parameters:
        -----------
        See sklearn.pipeline.Pipeline documentation.
        Note : all the steps must be pre-fitted / initialized.
        
        """
        super().__init__(steps, memory)

    def fit_transform(self, X, y=None):
        """Fit the model and transform with the final estimator.

        Process the sample seuqentially and for each fits all the 
        transforms one after the other and transforms the sample, 
        then uses fit_transform on the transformed data with the
        final estimator.

        Parameters:
        -----------
        X : iterable,
            Training data. Must fulfill input requirements of first
            step of the pipeline.
        
        Returns:
        --------
        Xt : pandas DataFrame, shape = [n_sample, n_transformed_features]
             Transformed samples.

        """
        try:
            #apply the whole pipeline fit_transform seqentially to all the sample
            output = pd.concat((super(SampleWisePipeline, self).fit_transform(sample) for sample in X), axis=0, join='outer')
            output = output.reset_index(drop=True)
        except AttributeError:
            print('One or multiple estimator in the pipeline are not pre-fitted / initialized.')
            raise

        return output
