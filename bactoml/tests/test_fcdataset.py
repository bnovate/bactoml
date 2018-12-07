import pytest

from datetime import datetime

from test_fixtures import fcs_path, locle_dir

from bactoml.fc_dataset import FCDataSet

class TestFCDataset(object):
    
    def test_load_dir(self, locle_dir):

        ds = FCDataSet(dir_path=locle_dir, sorted=True)

        #test the length of the dataset
        assert len(ds) == 10

        #test that the samples are sorted in time



ds = FCDataSet(dir_path=locle_dir, sorted=True)