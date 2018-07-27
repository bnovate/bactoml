#!/usr/bin/env python
# -*- coding: UTF8 -*-
"""
bactoml.test_data_load
======================

Tests data loading and dataset representation.

:author: Douglas Watson <douglas.watson@bnovate.com>
:date: 2018
:license: Private code, see LICENSE for more information

"""

import os
import glob
import pytest
import numpy as np

from test_fixtures import fcs_path, locle_dir

from bactoml.dataset import FCObservation, FCDataSet


class TestFCMObservation(object):

    def test_load_single(self, fcs_path):

        meas = FCObservation(fcs_path)

        assert meas.ID == "Cte8 31_05_2017 wac 30%"
        assert meas.datetime.year == 2017
        assert meas.datetime.month == 5

class TestFCMDataSet(object):

    def test_load_dir(self, locle_dir):
        paths = glob.glob(
            os.path.join(locle_dir, "*", "*.fcs")
        )
        ds = FCDataSet.from_paths(sorted(paths))

        assert len(ds.observations) == 10
        assert ds.observations[5].datetime.hour == 0  # CEST!
        assert list(ds.data.shape) == [10, 1]
