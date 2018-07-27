
#!/usr/bin/env python
# -*- coding: UTF8 -*-
"""
bactoml.preprocess
==================

Transformers used for pre-processing Flow Cytometry measurements.

:author: Douglas Watson <douglas.watson@bnovate.com>
:date: 2018
:license: Private code, see LICENSE for more information

"""

import numpy as np

from typing import Iterable
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

from . import dataset

MIN_FL1_RANGE = 10**1.5
MIN_FL2_RANGE = 10**1.5
MIN_SSC_RANGE = 10**1.5


class TLogTransformer(BaseEstimator, TransformerMixin):
    """ Truncated log transform for BactoSense results

    Parameters
    ----------
    min_ssc_range : float, optional
        Lower cut-off point for SSC events. Points under this line
        are projected up onto it.
    min_fl1_range : float, optional
    min_fl2_range : float, optional

    .. warning::

        These lower limits are given on the linear scale. Therefore,
        a lower limit of 3 on the log-log plot should be given as 10**3!

    Attributes
    ----------
    input_shape : tuple
        The shape the data passed to :meth:`fit`
    """

    def __init__(self, min_ssc_range=MIN_SSC_RANGE,
                 min_fl1_range=MIN_FL1_RANGE, min_fl2_range=MIN_FL2_RANGE):

        self.min_ssc_range = min_ssc_range
        self.min_fl1_range = min_fl1_range
        self.min_fl2_range = min_fl2_range

    def fit(self, X, y=None):
        """ Does nothing, included only to comply with sklearn API """
        return self

    def transform(self, X: Iterable[dataset.FCObservation]):
        """ A reference implementation of a transform function.

        Parameters
        ----------
        X : Array of FCObservation objects.

            Accepts sklearn standard where lines = observations,
            but not necessary. Each element of X is tlog transformed.

        Returns
        -------
        X_transformed : array of FCObservation objects

            Same shape as X.

        """
        # Input validation
        # X = check_array(X)

        # Create a copy of each FCObservation, and tlog transform it.
        X = np.array(X)
        transformed = []
        for obs in X.flatten():
            transformed.append(
                obs.copy().\
                    transform('tlog', channels=['SSC'], th=self.min_ssc_range, r=1, d=1, auto_range=False).\
                    transform('tlog', channels=['FL1'], th=self.min_fl1_range, r=1, d=1, auto_range=False).\
                    transform('tlog', channels=['FL2'], th=self.min_fl2_range, r=1, d=1, auto_range=False)
            )
        return np.array(transformed).reshape(X.shape)
