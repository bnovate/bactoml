
#!/usr/bin/env python
# -*- coding: UTF8 -*-
"""
bactoml.dataset
===============

Representations of data observations and datasets

Reference for how data sets should be represented:
http://scikit-learn.org/stable/tutorial/basic/tutorial.html#loading-an-example-dataset

:author: Douglas Watson <douglas.watson@bnovate.com>
:date: 2018
:license: Private code, see LICENSE for more information

"""

import datetime
import numpy as np
from typing import Iterable
from FlowCytometryTools.core.containers import FCMeasurement

class FCObservation(FCMeasurement):
    """ Represents a single FlowCytometry measurement.

    Currently subclasses FlowCytometryTools.FCMeasurement, but this could change
    in the future.

    Parameters
    ----------
    datafile : str Path to an FCS file

    Attributes
    ----------
    ID : str Name of the sample, taken from FCS file

    vol : float Analyzed volume (sample only, not including stain)

    datetime : datetime.datetime Measurement date

    date : datetime.date 

    time : datetime.time 

    Examples
    --------
    Load from an FCS file:

    >>> data = FCObservation("example.fcs") print data.vol 90.0


    Notes
    -----

    .. warning::

        In the case of FCS files saved from the BactoSense, datetime is 
        represented in the local timezone, not UTC. Therefore summer /
        winter time can lead to surprises.

    """

    def __init__(self, datafile, *args, **kwargs):
        super().__init__(ID="tmp", datafile=datafile, *args, **kwargs)

        self.ID = self.meta['$SMNO']

        date_string = "{$DATE} {$BTIM}".format(**self.meta)
        self.datetime = datetime.datetime.strptime(date_string, "%d-%b-%Y %H:%M:%S")
        self.date = self.datetime.date()
        self.time = self.datetime.time()
        self.vol = float(self.meta['$VOL']) / 1000.

    def plot(self, fig=None):
        # TODO: plot FL1-FL2, FL1-SSC
        pass


class FCDataSet(object):
    """ Represents a labelled set of FCObservations 

    Parameters
    ----------
    observations : array of FCObservations

        Vertical array with one observation per *line*, following sklearn
        standards.

    labels : list of str, list of bool, list of float...

        Can be None, if the dataset is unlabelled.

    Examples
    --------
    Load a dataset for a list of FCS file paths:

    >>> paths = glob.glob("*/*.fcs")
    >>> dataset = FCDataSet.from_paths(paths)

    Set labels:

    >>> dataset.labels = my_labels
    
    """

    def __init__(self, observations: Iterable, labels: Iterable=None):
        #: sklearn compliant representation:
        self.data = np.array(observations).reshape(-1, 1) 
        self.labels = labels

    @classmethod
    def from_paths(cls, paths: Iterable[str], labels=None):
        """ Load dataset from a folder. 

        Parameters
        ----------
        paths : list of str

        labels : list of str, optional
            labels sorted in the same order as paths

        Examples
        --------

        >>> paths = sorted(glob.glob("*/*.fcs"))
        >>> dataset = FCDataSet.from_paths(paths)
        
        """
        obs = [FCObservation(p) for p in paths]
        return FCDataSet(obs, labels)

    @property
    def observations(self):
        """ Provides a flat view of self.data 
        
        Instead of looking up data[i, 1], look up observations[i]
        """
        return self.data.flatten()

    def plot_timeseries(self):
        """ Show TCC / HNA over time """
        pass

    def animate(self):
        """ Display dotplots with slider to animate.

        For use in jupyter notebooks.
        """
        pass
