import os
import pytest
import numpy as np
import datetime
import itertools
import pathlib

from test_fixtures import fcs_path, locle_dir

import FlowCytometryTools

from bactoml.fcdataset import FCDataSet, MissingPathError

"""import sys
sys.path.insert(0, '.')
from fcdataset import FCDataSet, MissingPathError"""

class TestFCDataset(object):
    
    def test_load_dir(self, locle_dir):

        ds = FCDataSet(dir_path=locle_dir, sorted=True)

        #test the length of the dataset
        assert len(ds) == 10

        #test exception for missing directory
        with pytest.raises(MissingPathError):
            FCDataSet(os.path.join(locle_dir, os.path.pardir, 'missing_dir'))

    def test_sort(self, locle_dir):

        ds = FCDataSet(dir_path=locle_dir, sorted=True)

        #test that the samples are sorted in time
        assert all(np.diff([datetime.datetime.strptime("{$DATE} {$BTIM}".format(**fcs.get_meta()), "%d-%b-%Y %H:%M:%S") for fcs in ds]) >= datetime.timedelta(0))

    def test_getitem(self, locle_dir):

        ds = FCDataSet(dir_path=locle_dir, sorted=True)

        #test FCMeasurement instanciation at indexing moment
        assert isinstance(ds[0], FlowCytometryTools.FCMeasurement)

        #test out of range index
        with pytest.raises(IndexError):
            ds[11]

        #test negative index behavior
        assert ds[-1].get_data().equals(ds[len(ds)-1].get_data())

        #test type error for index
        with pytest.raises(TypeError): 
            ds['a']
        
        #test dataset slicing
        ds_slice = ds[2:5]
        assert all([fcs.get_data().equals(ds[i].get_data()) for fcs, i in zip(ds_slice, itertools.count(2))])


    def test_setitem(self, locle_dir, fcs_path):

        ds = FCDataSet(dir_path=locle_dir, sorted=True)

        #test setitem
        ds[0] = FlowCytometryTools.FCMeasurement(ID='blank', datafile=fcs_path)
        assert pathlib.Path(ds[0].datafile).samefile(pathlib.Path(fcs_path))

        #test setitem with slice
        ds[2:4] = list(itertools.repeat(FlowCytometryTools.FCMeasurement(ID='blank', datafile=fcs_path), 2))
        assert pathlib.Path(ds[3].datafile).samefile(pathlib.Path(fcs_path))

        #test type error for input
        with pytest.raises(TypeError):
            ds[0] = 1.2

        #test missing fcs file
        with pytest.raises(MissingPathError):
            ds[0] = 'missing_file.fcs'

        #test type error for index
        with pytest.raises(TypeError):
            ds[1.2] = FlowCytometryTools.FCMeasurement(ID='blank', datafile=fcs_path)

        #test out of range index
        with pytest.raises(IndexError):
            ds[11] = FlowCytometryTools.FCMeasurement(ID='blank', datafile=fcs_path)


    def test_insert(self, locle_dir, fcs_path):

        ds = FCDataSet(dir_path=locle_dir, sorted=True)
        l0 = len(ds)

        #test insertion
        ds.insert(0, FlowCytometryTools.FCMeasurement(ID='blank', datafile=fcs_path))
        assert len(ds) == (l0 + 1)
        assert pathlib.Path(ds[0].datafile).samefile(pathlib.Path(fcs_path))

        #test type error for input
        with pytest.raises(TypeError):
            ds.insert(0, 1.2)

        #test missing fcs file
        with pytest.raises(MissingPathError):
            ds.insert(0, 'missing_file.fcs')

        #test type error for index
        with pytest.raises(TypeError):
            ds.insert(1.2, FlowCytometryTools.FCMeasurement(ID='blank', datafile=fcs_path))

    def test_delitem(self, locle_dir):
        
        ds = FCDataSet(dir_path=locle_dir, sorted=True)
        l0 = len(ds)
        ds_1 = ds[1]

        #test deltion of one item
        del ds[0]
        assert len(ds) == l0 - 1
        assert pathlib.Path(ds[0].datafile).samefile(pathlib.Path(ds_1.datafile))

        #test deletion of slices
        del ds[0:len(ds)]
        assert len(ds) == 0
        with pytest.raises(IndexError):
            del ds[0]