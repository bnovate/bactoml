#!/usr/bin/env python
# -*- coding: UTF8 -*-
"""
bactoml.test_preprocess
=======================

Test data transformers.

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
from bactoml.preprocess import TLogTransformer

class TestTlogTransform(object):

    def test_transform_single(self, fcs_path):
        """ Transforming single results should yield tlog10 """

        meas = FCObservation(fcs_path)

        # Currently on a linear scale, values up to 10**6 or higher
        assert meas.data['FL1'].max() > 10

        tlog = TLogTransformer(min_fl2_range=10**3, min_ssc_range=10**2)

        raw = [meas]
        tlog.fit(raw)
        transformed = tlog.transform(raw)

        assert len(transformed) == 1

        mt = transformed[0]
        # Original should not be changed:
        assert meas.data['FL1'].max() > 10

        # Modified data should lie within the tlog range:
        assert mt.data['FL1'].max() < 10
        assert mt.data['FL2'].min() >= 2.99  # Some wiggle room...
        assert mt.data['SSC'].min() >= 1.99  # ...for float errors


    def test_transform_multiple(self, locle_dir):
        paths = glob.glob(
            os.path.join(locle_dir, "*", "*.fcs")
        )
        ds = FCDataSet.from_paths(sorted(paths))

        tlog = TLogTransformer(min_fl2_range=10**3, min_ssc_range=10**2)
        tlog.fit(ds.data)

        transf = tlog.transform(ds.data)
        assert list(transf.shape) == [10, 1]